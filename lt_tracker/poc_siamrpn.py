import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
from ultralytics import FastSAM
import os, time, csv, queue, threading, subprocess
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple
import sys


@dataclass
class Config:
    project_dir: str = "."
    input_video: str = "coyote.mp4"
    base_output_dir: str = field(default="")
    num_frames: int = 2001
    init_bbox: Tuple[int, int, int, int] = (695, 345, 20, 30)
    
    # Axelera
    num_tile_workers: int = 3
    aipu_cores_per_worker: int = 1
    
    # Sizes
    exemplar_size: int = 127
    instance_size: int = 351
    lost_instance_size: int = 650
    num_tiles: int = 2
    
    # Tracking params
    penalty_k: float = 0.05
    window_influence: float = 0.35
    lr: float = 0.28
    conf_low: float = 0.8
    conf_high: float = 0.985
    enable_lt_mode: bool = True
    
    # Anchors
    anchor_stride: int = 8
    anchor_ratios: Tuple = (0.33, 0.5, 1, 2, 3)
    anchor_scales: Tuple = (8,)
    base_size: int = 8
    context_amount: float = 0.5
    
    # Visualization
    inset_width: int = 150
    inset_padding: int = 10
    frame_export_interval: int = 100
    show_search_crop: bool = True
    timing_report_interval: int = 100
    
    def __post_init__(self):
        self.base_output_dir = self.base_output_dir or os.path.join(self.project_dir, "experiments")
        self.tile_size = self.instance_size
    
    @property
    def search_encoder_ax(self):
        base = os.path.join(self.project_dir, 'build')
        suffix = 'siamrpn++onnx' if self.instance_size == 351 else 'siamrpn++onnx_255'
        return f'{base}/{suffix}/{suffix}/1/model.json'
    
    @property
    def head_onnx(self):
        name = 'siamrpn_head_37.onnx' if self.instance_size == 351 else 'siamrpn_head_simp.onnx'
        return os.path.join(self.project_dir, name)
    
    @property
    def config_path(self):
        return os.path.join(self.project_dir, 'pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml')
    
    @property
    def model_path(self):
        return os.path.join(self.project_dir, 'model.pth')


CFG = Config()

sys.path.append(os.path.join(CFG.project_dir, 'pysot'))
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder

# --- Utilities ---

def create_experiment_folder(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing = [int(d.split('_')[1]) for d in os.listdir(base_dir) 
                if d.startswith('exp_') and os.path.isdir(os.path.join(base_dir, d))]
    exp_num = max(existing, default=0) + 1
    folder = f"exp_{exp_num:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_path = os.path.join(base_dir, folder)
    frames_path = os.path.join(exp_path, "frames")
    os.makedirs(frames_path, exist_ok=True)
    return exp_path, frames_path, exp_num


def write_settings(exp_path, cfg):
    with open(os.path.join(exp_path, "settings.txt"), 'w') as f:
        f.write(f"SiamRPN LT Tracker Settings\nGenerated: {datetime.now()}\n\n")
        for k, v in cfg.__dict__.items():
            f.write(f"{k} = {v}\n")


def convert_video(inp, out):
    cmd = ['ffmpeg', '-y', '-i', inp, '-c:v', 'libx264', '-crf', '23', '-preset', 'medium', '-c:a', 'aac', '-b:a', '128k', out]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Converted: {out}")
    except Exception as e:
        print(f"FFmpeg error: {e}")


@dataclass
class TileJob:
    idx: int
    x: int
    y: int
    data: np.ndarray

@dataclass 
class TileResult:
    idx: int
    x: int
    y: int
    features: List[np.ndarray]
    time_ms: float


class TimingStats:
    STAGES = ['preprocess', 'backbone', 'dw_xcorr', 'head', 'postprocess', 'total']
    
    def __init__(self):
        self.times = defaultdict(list)
    
    def add(self, stage, ms):
        self.times[stage].append(ms)
    
    def report(self, frame_num=None):
        print(f"\n{'='*70}\nTiming Report{f' (frame {frame_num})' if frame_num else ''}\n{'='*70}")
        print(f"{'Stage':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Count':>8}")
        for stage in self.STAGES:
            if not self.times[stage]: continue
            arr = np.array(self.times[stage])
            print(f"{stage:<15} {arr.mean():>10.3f} {arr.std():>10.3f} {arr.min():>10.3f} {arr.max():>10.3f} {len(arr):>8}")
        if self.times['total']:
            print(f"\nAverage FPS: {1000.0 / np.mean(self.times['total']):.2f}")


# --- Axelera Quantization Mixin ---

class AxeleraQuantMixin:
    def _init_quant(self, model):
        self.input_infos = model.inputs()
        self.output_infos = model.outputs()
        self.input_info = self.input_infos[0]
        self.input_buffers = [np.zeros(t.shape, np.int8) for t in self.input_infos]
        self.output_buffers = [np.zeros(t.shape, np.int8) for t in self.output_infos]
    
    def _quantize(self, img):
        info = self.input_info
        if img.ndim == 4: img = img.transpose(0, 2, 3, 1).squeeze(0)
        elif img.ndim == 3 and img.shape[0] == 3: img = img.transpose(1, 2, 0)
        q = np.clip(np.round(img / info.scale + info.zero_point), -128, 127).astype(np.int8)
        if info.padding and any(p != (0, 0) for p in info.padding):
            q = np.pad(q, info.padding[1:], constant_values=info.zero_point)
        return q
    
    def _dequantize(self, idx):
        info = self.output_infos[idx]
        out = self.output_buffers[idx]
        if info.padding and any(p != (0, 0) for p in info.padding):
            out = out[tuple(slice(b, -e or None) for b, e in info.padding)]
        result = (out.astype(np.float32) - info.zero_point) * info.scale
        return result.transpose(0, 3, 1, 2) if result.ndim == 4 else result


class AxeleraInferenceWorker(AxeleraQuantMixin):
    def __init__(self, ctx, model, worker_id, aipu_cores=2):
        self.worker_id = worker_id
        self.inqueue, self.outqueue = queue.Queue(4), queue.Queue(4)
        self.running = True
        self._init_quant(model)
        conn = ctx.device_connect(None, 1)
        self.instance = conn.load_model_instance(model, num_sub_devices=1, aipu_cores=aipu_cores)
        self._thread = threading.Thread(target=self._run, name=f"TileWorker-{worker_id}")
        self._thread.start()
    
    def _run(self):
        while self.running:
            try:
                job = self.inqueue.get(timeout=0.1)
                if job is None: break
                t0 = time.perf_counter()
                self.input_buffers[0][:] = self._quantize(job.data)
                self.instance.run(self.input_buffers, self.output_buffers)
                outputs = [self._dequantize(i).copy() for i in range(len(self.output_infos))]
                self.outqueue.put(TileResult(job.idx, job.x, job.y, outputs, (time.perf_counter() - t0) * 1000))
            except queue.Empty:
                continue
    
    def stop(self):
        self.running = False
        self.inqueue.put(None)
        self._thread.join(timeout=5.0)


class AxeleraSearchEncoder(AxeleraQuantMixin):
    def __init__(self, ctx, model, aipu_cores=4):
        self._init_quant(model)
        conn = ctx.device_connect(None, 1)
        self.instance = conn.load_model_instance(model, num_sub_devices=1, aipu_cores=aipu_cores)
    
    def run(self, img):
        t0 = time.perf_counter()
        self.input_buffers[0][:] = self._quantize(img)
        self.instance.run(self.input_buffers, self.output_buffers)
        outputs = [self._dequantize(i) for i in range(len(self.output_infos))]
        return outputs, (time.perf_counter() - t0) * 1000


class XCorrDepthwise:
    def __call__(self, search, kernel):
        results = []
        for i in range(6):
            s, k = torch.from_numpy(search[i]), torch.from_numpy(kernel[i]).transpose(0, 1)
            results.append(F.conv2d(s, k, groups=s.shape[1]))
        return torch.cat(results, dim=1).numpy()


class TemplateEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.neck = getattr(model, 'neck', None)
        self.cls_kernels = nn.ModuleList([rpn.cls.conv_kernel for rpn in [model.rpn_head.rpn2, model.rpn_head.rpn3, model.rpn_head.rpn4]])
        self.loc_kernels = nn.ModuleList([rpn.loc.conv_kernel for rpn in [model.rpn_head.rpn2, model.rpn_head.rpn3, model.rpn_head.rpn4]])
    
    def forward(self, x):
        zf = self.backbone(x)
        if self.neck: zf = self.neck(zf)
        zf = list(zf.values()) if isinstance(zf, dict) else zf
        return [out for i, z in enumerate(zf) for out in (self.cls_kernels[i](z), self.loc_kernels[i](z))]


class SiamRPNLTTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = 'cpu'
        
        from axelera.runtime import Context
        ctx = Context()
        model = ctx.load_model(cfg.search_encoder_ax)
        
        total_cores = cfg.num_tile_workers * cfg.aipu_cores_per_worker
        self.primary_encoder = AxeleraSearchEncoder(ctx, model, aipu_cores=total_cores)
        self.tile_workers = [AxeleraInferenceWorker(ctx, model, i, cfg.aipu_cores_per_worker) 
                            for i in range(cfg.num_tile_workers)]
        
        self.head_session = ort.InferenceSession(cfg.head_onnx, providers=['CPUExecutionProvider'])
        
        cfgp = cfg.config_path
        cfg_obj = cfg
        from pysot.core.config import cfg
        cfg.merge_from_file(cfgp)
        pth_model = ModelBuilder()
        pth_model.load_state_dict(torch.load(cfg_obj.model_path, map_location='cpu'))
        self.template_encoder = TemplateEncoder(pth_model).to(self.device).eval()
        
        self.xcorr = XCorrDepthwise()
        self.anchor_num = len(cfg_obj.anchor_ratios) * len(cfg_obj.anchor_scales)
        self.timing = TimingStats()
        self.frame_num = 0
        self._reset_state()
    
    def _reset_state(self):
        self.center_pos = self.size = self.channel_avg = None
        self.kernel_np = None
        self.longterm = False
        self.last_crop = self.last_crop_box = None
    
    def _generate_anchors(self, score_size):
        cfg = self.cfg
        anchors = np.zeros((self.anchor_num, 4))
        size = cfg.anchor_stride ** 2
        idx = 0
        for r in cfg.anchor_ratios:
            ws = int(np.sqrt(size / r))
            hs = int(ws * r)
            for s in cfg.anchor_scales:
                anchors[idx] = [-ws*s/2, -hs*s/2, ws*s/2, hs*s/2]
                idx += 1
        anchor = np.zeros((self.anchor_num, score_size, score_size, 4))
        ori = -(score_size // 2) * cfg.anchor_stride
        for i in range(score_size):
            for j in range(score_size):
                x, y = ori + j * cfg.anchor_stride, ori + i * cfg.anchor_stride
                anchor[:, i, j, :] = anchors + [x, y, x, y]
        return anchor.reshape(-1, 4)
    
    def _get_subwindow(self, img, pos, model_sz, orig_sz):
        im_h, im_w = img.shape[:2]
        c = (orig_sz + 1) / 2
        xmin, xmax = int(pos[0] - c), int(pos[0] - c) + orig_sz
        ymin, ymax = int(pos[1] - c), int(pos[1] - c) + orig_sz
        self.last_crop_box = (pos[0] - orig_sz/2, pos[1] - orig_sz/2, orig_sz, orig_sz)
        
        lp, tp = max(0, -xmin), max(0, -ymin)
        rp, bp = max(0, xmax - im_w), max(0, ymax - im_h)
        xmin, xmax, ymin, ymax = xmin + lp, xmax + lp, ymin + tp, ymax + tp
        
        if any([tp, bp, lp, rp]):
            padded = np.full((im_h + tp + bp, im_w + lp + rp, 3), self.channel_avg, dtype=np.uint8)
            padded[tp:tp+im_h, lp:lp+im_w] = img
            patch = padded[ymin:ymax, xmin:xmax]
        else:
            patch = img[ymin:ymax, xmin:xmax]
        
        if model_sz != orig_sz:
            patch = cv2.resize(patch, (model_sz, model_sz))
        self.last_crop = patch.copy()
        return patch
    
    def _convert_score(self, cls, score_size):
        cls = cls.squeeze().reshape(2, self.anchor_num, score_size, score_size).reshape(2, -1).T
        e_x = np.exp(cls - cls.max(axis=1, keepdims=True))
        return (e_x / e_x.sum(axis=1, keepdims=True))[:, 1]
    
    def _convert_bbox(self, loc, anchors, score_size):
        loc = loc.squeeze().reshape(4, self.anchor_num, score_size, score_size).reshape(4, -1)
        aw, ah = anchors[:, 2] - anchors[:, 0], anchors[:, 3] - anchors[:, 1]
        acx, acy = (anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2
        return np.vstack([loc[0] * aw + acx, loc[1] * ah + acy, np.exp(loc[2]) * aw, np.exp(loc[3]) * ah])
    
    def _track_tiled(self, img, s_x, scale_z):
        cfg = self.cfg
        full_size = round(s_x)
        crop = self._get_subwindow(img, self.center_pos, full_size, full_size)
        
        stride = (full_size - cfg.tile_size) // (cfg.num_tiles - 1) if cfg.num_tiles > 1 else 0
        tiles = [(min(j * stride, full_size - cfg.tile_size), min(i * stride, full_size - cfg.tile_size))
                 for i in range(cfg.num_tiles) for j in range(cfg.num_tiles)]
        
        tile_score_size = (cfg.tile_size - cfg.exemplar_size) // cfg.anchor_stride + 1 + cfg.base_size
        anchors = self._generate_anchors(tile_score_size)
        
        # Submit jobs
        for idx, (tx, ty) in enumerate(tiles):
            tile = crop[ty:ty+cfg.tile_size, tx:tx+cfg.tile_size]
            if tile.shape[:2] != (cfg.tile_size, cfg.tile_size):
                padded = np.full((cfg.tile_size, cfg.tile_size, 3), self.channel_avg, dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            self.tile_workers[idx % cfg.num_tile_workers].inqueue.put(
                TileJob(idx, tx, ty, np.expand_dims(tile.astype(np.float32).transpose(2, 0, 1), 0)))
        
        # Collect results
        t0 = time.perf_counter()
        results = []
        for _ in tiles:
            for w in self.tile_workers:
                try:
                    results.append(w.outqueue.get(timeout=0.01))
                    break
                except queue.Empty:
                    continue
        backbone_time = (time.perf_counter() - t0) * 1000
        results.sort(key=lambda r: r.idx)
        
        all_scores, all_bboxes = [], []
        for r in results:
            xcorr = self.xcorr(r.features, self.kernel_np)
            cls, loc = self.head_session.run(None, {'xcorr_features': xcorr})
            score = self._convert_score(cls, tile_score_size)
            bbox = self._convert_bbox(loc, anchors, tile_score_size)
            bbox[0] += r.x + cfg.tile_size / 2 - full_size / 2
            bbox[1] += r.y + cfg.tile_size / 2 - full_size / 2
            all_scores.append(score)
            all_bboxes.append(bbox)
        
        self.timing.add('backbone', backbone_time)
        return np.concatenate(all_scores), np.hstack(all_bboxes), np.ones(len(np.concatenate(all_scores))) * 0.5
    
    def init(self, img, bbox):
        cfg = self.cfg
        x, y, w, h = bbox
        self.center_pos = np.array([x + w/2, y + h/2])
        self.size = np.array([w, h])
        wz = w + cfg.context_amount * (w + h)
        hz = h + cfg.context_amount * (w + h)
        s_z = round(np.sqrt(wz * hz))
        self.channel_avg = np.mean(img, axis=(0, 1))
        
        z_crop = self._get_subwindow(img, self.center_pos, cfg.exemplar_size, s_z)
        z_input = torch.from_numpy(z_crop.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            kernel = self.template_encoder(z_input)
        self.kernel_np = [k.cpu().numpy() for k in kernel]
        self.longterm = False
    
    def track(self, img):
        cfg = self.cfg
        t_start = time.perf_counter()
        
        w, h = self.size
        wz = w + cfg.context_amount * (w + h)
        hz = h + cfg.context_amount * (w + h)
        s_z = np.sqrt(wz * hz)
        scale_z = cfg.exemplar_size / s_z
        inst_size = cfg.lost_instance_size if self.longterm else cfg.instance_size
        s_x = s_z * (inst_size / cfg.exemplar_size)
        
        if self.longterm:
            s_x = max(s_x, max(cfg.lost_instance_size, cfg.tile_size * cfg.num_tiles * 0.75))
            scale_z = 1.0
            score, bbox, window = self._track_tiled(img, s_x, scale_z)
        else:
            score_size = (inst_size - cfg.exemplar_size) // cfg.anchor_stride + 1 + cfg.base_size
            hann = np.hanning(score_size)
            window = np.tile(np.outer(hann, hann).flatten(), self.anchor_num)
            anchors = self._generate_anchors(score_size)
            
            t0 = time.perf_counter()
            x_crop = self._get_subwindow(img, self.center_pos, inst_size, round(s_x))
            x_input = np.expand_dims(x_crop.astype(np.float32).transpose(2, 0, 1), 0)
            self.timing.add('preprocess', (time.perf_counter() - t0) * 1000)
            
            features, bb_time = self.primary_encoder.run(x_input)
            self.timing.add('backbone', bb_time)
            
            t0 = time.perf_counter()
            xcorr = self.xcorr(features, self.kernel_np)
            self.timing.add('dw_xcorr', (time.perf_counter() - t0) * 1000)
            
            t0 = time.perf_counter()
            cls, loc = self.head_session.run(None, {'xcorr_features': xcorr})
            self.timing.add('head', (time.perf_counter() - t0) * 1000)
            
            t0 = time.perf_counter()
            score = self._convert_score(cls, score_size)
            bbox = self._convert_bbox(loc, anchors, score_size)
            self.timing.add('postprocess', (time.perf_counter() - t0) * 1000)
        
        self.frame_num += 1
        
        # Penalties and scoring
        def change(r): return np.maximum(r, 1. / r)
        def sz(w, h): pad = (w + h) * 0.5; return np.sqrt((w + pad) * (h + pad))
        
        s_c = change(sz(bbox[2], bbox[3]) / sz(self.size[0] * scale_z, self.size[1] * scale_z))
        r_c = change((self.size[0] / self.size[1]) / (bbox[2] / bbox[3]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.penalty_k)
        pscore = penalty * score
        win_inf = 0.001 if self.longterm else cfg.window_influence
        pscore = pscore * (1 - win_inf) + window * win_inf
        
        best_idx = np.argmax(pscore)
        best_bbox = bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.lr
        best_score = score[best_idx]
        
        if best_score >= cfg.conf_low:
            cx = best_bbox[0] + self.center_pos[0]
            cy = best_bbox[1] + self.center_pos[1]
            width = self.size[0] * (1 - lr) + best_bbox[2] * lr
            height = self.size[1] * (1 - lr) + best_bbox[3] * lr
        else:
            cx, cy, width, height = *self.center_pos, *self.size
        
        im_h, im_w = img.shape[:2]
        self.center_pos = np.array([np.clip(cx, 0, im_w), np.clip(cy, 0, im_h)])
        self.size = np.array([np.clip(width, 10, im_w), np.clip(height, 10, im_h)])
        
        if cfg.enable_lt_mode:
            self.longterm = best_score < cfg.conf_low if best_score < cfg.conf_low else (False if best_score > cfg.conf_high else self.longterm)
        
        self.timing.add('total', (time.perf_counter() - t_start) * 1000)
        return {'bbox': [cx - width/2, cy - height/2, width, height], 'score': float(best_score), 'longterm': self.longterm}
    
    def stop(self):
        for w in self.tile_workers:
            w.stop()


# --- Visualization ---

def add_inset(frame, bbox, width=150, pad=10):
    bx, by, bw, bh = [max(0, int(v)) for v in bbox]
    bw, bh = min(bw, frame.shape[1] - bx), min(bh, frame.shape[0] - by)
    if bw <= 0 or bh <= 0: return frame
    
    scale = width / bw
    ih = min(int(bh * scale), frame.shape[0] - 2 * pad)
    if ih <= 0: return frame
    iw = int(bw * (ih / bh)) if ih < int(bh * scale) else width
    
    inset = cv2.resize(frame[by:by+bh, bx:bx+bw], (iw, ih))
    cv2.rectangle(inset, (0, 0), (iw-1, ih-1), (255, 255, 255), 2)
    frame[frame.shape[0]-ih-pad:frame.shape[0]-pad, pad:pad+iw] = inset
    return frame


def draw_frame(frame, result, frame_num, total, tracker, cfg):
    x, y, w, h = result['bbox']
    score, longterm = result['score'], result['longterm']
    
    frame = add_inset(frame, (x, y, w, h), cfg.inset_width, cfg.inset_padding)
    
    if cfg.show_search_crop and tracker.last_crop_box:
        cx, cy, cw, ch = tracker.last_crop_box
        color = (0, 165, 255) if longterm else (0, 255, 255)
        cv2.rectangle(frame, (int(cx), int(cy)), (int(cx+cw), int(cy+ch)), color, 2)
    
    color = (0, 165, 255) if longterm else ((0, 255, 255) if score > cfg.conf_low else (0, 0, 255))
    status = "SEARCHING" if longterm else ("TRACKING" if score > cfg.conf_low else "LOW_CONF")
    
    cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
    label = f"{status} ({score:.2f})"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (int(x), int(y)-lh-10), (int(x)+lw, int(y)), color, -1)
    cv2.putText(frame, label, (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.circle(frame, (int(x + w/2), int(y + h/2)), 4, color, -1)
    
    texts = [(f"Frame: {frame_num}/{total}", (10, 30)), (f"Score: {score:.3f}", (10, 60)),
             (f"LT: {'ON' if longterm else 'OFF'}", (10, 120))]
    for txt, pos in texts:
        cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def refine_bbox_with_fastsam(frame, bbox, fastsam):
    x, y, w, h = bbox
    crop = frame[y:y+h, x:x+w]
    results = fastsam(crop, device='cpu', retina_masks=True, conf=0.3, iou=0.9)
    
    if not results or not results[0].masks: return bbox, results
    
    best_area, refined = 0, bbox
    for mask in results[0].masks.data:
        m = mask.cpu().numpy().astype(bool)
        if m.shape != (h, w): m = cv2.resize(m.astype(np.uint8), (w, h)).astype(bool)
        ys, xs = np.where(m)
        if len(xs) > 0:
            x1, y1, x2, y2 = xs.min() + x, ys.min() + y, xs.max() + x, ys.max() + y
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area, refined = area, (x1, y1, x2 - x1, y2 - y1)
    return refined, results


def main():
    cfg = CFG
    exp_path, frames_path, exp_num = create_experiment_folder(cfg.base_output_dir)
    print(f"Experiment {exp_num}: {exp_path}")
    write_settings(exp_path, cfg)
    
    tracker = SiamRPNLTTracker(cfg)
    fastsam = FastSAM("FastSAM-x.pt")
    
    cap = cv2.VideoCapture(cfg.input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = min(cfg.num_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    out = cv2.VideoWriter(os.path.join(exp_path, "output.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    csv_file = open(os.path.join(exp_path, "tracking_results.csv"), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'state', 'x', 'y', 'w', 'h', 'conf'])
    
    try:
        ret, frame = cap.read()
        refined, _ = refine_bbox_with_fastsam(frame, cfg.init_bbox, fastsam)
        tracker.init(frame, refined)
        
        x, y, bw, bh = refined
        csv_writer.writerow([0, 'INIT', x, y, bw, bh, 1.0])
        
        display = frame.copy()
        cv2.rectangle(display, (int(x), int(y)), (int(x+bw), int(y+bh)), (0, 255, 0), 2)
        out.write(display)
        cv2.imwrite(os.path.join(frames_path, "frame_00000.jpg"), display)
        
        for frame_num in range(1, total):
            ret, frame = cap.read()
            if not ret: break
            
            result = tracker.track(frame)
            x, y, bw, bh = result['bbox']
            state = "SEARCHING" if result['longterm'] else ("TRACKING" if result['score'] > cfg.conf_low else "LOW_CONF")
            csv_writer.writerow([frame_num, state, round(x, 2), round(y, 2), round(bw, 2), round(bh, 2), round(result['score'], 4)])
            
            display = draw_frame(frame.copy(), result, frame_num, total, tracker, cfg)
            out.write(display)
            
            if frame_num % cfg.frame_export_interval == 0:
                cv2.imwrite(os.path.join(frames_path, f"frame_{frame_num:05d}.jpg"), display)
            if frame_num % 100 == 0:
                print(f"Processed {frame_num}/{total}")
            if frame_num % cfg.timing_report_interval == 0:
                tracker.timing.report(frame_num)
    finally:
        tracker.stop()
        cap.release()
        out.release()
        csv_file.close()
    
    tracker.timing.report()
    convert_video(os.path.join(exp_path, "output.mp4"), os.path.join(exp_path, "output_converted.mp4"))
    print(f"\nExperiment {exp_num} complete! Outputs: {exp_path}")


if __name__ == '__main__':
    main()