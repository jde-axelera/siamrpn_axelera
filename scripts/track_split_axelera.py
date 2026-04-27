#!/usr/bin/env python3
"""
track_split_axelera.py — SiamRPN++ tracker with Axelera Metis backbone
=======================================================================

Inference pipeline per frame:
  [Metis] template_encoder  (1×, at init / template update)
  [Metis] search_encoder    (every frame)
  [CPU]   xcorr_head.onnx   (every frame — dynamic-weight grouped conv)
  [CPU]   post-processing + particle filter

Usage:
    python scripts/track_split_axelera.py --config configs/track_config_axelera.json

Config keys (paths):
    pysot_root           — directory containing the pysot/ sub-folder
    model_cfg            — config_ir_siamese_infer.yaml
    checkpoint           — best_model.pth
    template_encoder_ax  — compiled_template/compiled_model/model.json
    search_encoder_ax    — compiled_search/compiled_model/model.json
    xcorr_head           — xcorr_head.onnx  (CPU)
    video                — input video
    sam2_video           — optional SAM2 panel video (omit key to skip)
    output               — output mp4 path
"""

import argparse, json, os, sys, time, types, subprocess
import numpy as np
import cv2
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(path):
    with open(path) as f:
        return json.load(f)


# ── Constants ─────────────────────────────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)


# ── Helpers (same as track_split.py) ─────────────────────────────────────────

def systematic_resample(weights):
    N = len(weights)
    weights = weights / (weights.sum() + 1e-300)
    positions = (np.random.random() + np.arange(N)) / N
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    i, j = 0, 0
    indices = np.zeros(N, dtype=int)
    while i < N:
        if positions[i] < cumsum[j]:
            indices[i] = j; i += 1
        else:
            j = min(j + 1, N - 1)
    return indices


def cls_to_scoremap(cls_raw, n_anchors=5, size=25):
    s = cls_raw.transpose(1, 2, 3, 0).reshape(2, -1).T
    e = np.exp(s - s.max(axis=1, keepdims=True))
    fg = (e / e.sum(axis=1, keepdims=True))[:, 1]
    return fg, fg.reshape(n_anchors, size, size).max(axis=0)


def particle_to_feat(px, py, cx_search, cy_search, s_x):
    scale = 255.0 / (s_x + 1e-6)
    sx_p = (px - cx_search) * scale + 127.5
    sy_p = (py - cy_search) * scale + 127.5
    return (np.clip(((sx_p - 31.5) / 8.0).astype(int), 0, 24),
            np.clip(((sy_p - 31.5) / 8.0).astype(int), 0, 24))


def add_label(img, text, bar_h=20):
    bar = np.full((bar_h, img.shape[1], 3), (15, 15, 15), dtype=np.uint8)
    cv2.putText(bar, text, (4, bar_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def put(img, txt, x, y, scale=0.42, col=(200, 200, 200)):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def fit_panel(img, target_h, panel_w):
    oh, ow = img.shape[:2]
    new_w = int(round(ow * target_h / oh))
    img = cv2.resize(img, (new_w, target_h))
    if new_w == panel_w:
        return img
    elif new_w > panel_w:
        x0 = (new_w - panel_w) // 2
        return img[:, x0:x0 + panel_w]
    pad = np.zeros((target_h, panel_w, 3), dtype=np.uint8)
    pad[:, (panel_w - new_w) // 2:(panel_w - new_w) // 2 + new_w] = img
    return pad


def decode_search(x_np):
    img = np.clip((x_np[0].transpose(1, 2, 0) * STD + MEAN) * 255, 0, 255).astype(np.uint8)
    return img[:, :, ::-1].copy()


# ── Axelera inference wrappers ────────────────────────────────────────────────

class AxeleraQuantMixin:
    def _init_quant(self, model):
        self.input_infos  = model.inputs()
        self.output_infos = model.outputs()
        self.input_info   = self.input_infos[0]
        self.input_buffers  = [np.zeros(t.shape, np.int8) for t in self.input_infos]
        self.output_buffers = [np.zeros(t.shape, np.int8) for t in self.output_infos]

    def _quantize(self, img):
        info = self.input_info
        if img.ndim == 4:
            img = img.transpose(0, 2, 3, 1).squeeze(0)    # NCHW → HWC
        elif img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)                   # CHW → HWC
        q = np.clip(np.round(img / info.scale + info.zero_point), -128, 127).astype(np.int8)
        if info.padding and any(p != (0, 0) for p in info.padding):
            q = np.pad(q, info.padding[1:], constant_values=info.zero_point)
        return q

    def _dequantize(self, idx):
        info = self.output_infos[idx]
        out  = self.output_buffers[idx]
        if info.padding and any(p != (0, 0) for p in info.padding):
            out = out[tuple(slice(b, -e or None) for b, e in info.padding)]
        result = (out.astype(np.float32) - info.zero_point) * info.scale
        return result.transpose(0, 3, 1, 2) if result.ndim == 4 else result


class AxeleraEncoder(AxeleraQuantMixin):
    """Single-shot encoder (template or search) on Axelera Metis."""
    def __init__(self, ctx, model, aipu_cores=4):
        self._init_quant(model)
        conn = ctx.device_connect(None, 1)
        self.instance = conn.load_model_instance(model, num_sub_devices=1, aipu_cores=aipu_cores)

    def run(self, img_nchw):
        """img_nchw: float32 (1, 3, H, W). Returns (list[ndarray], latency_ms)."""
        t0 = time.perf_counter()
        self.input_buffers[0][:] = self._quantize(img_nchw)
        self.instance.run(self.input_buffers, self.output_buffers)
        outputs = [self._dequantize(i) for i in range(len(self.output_infos))]
        return outputs, (time.perf_counter() - t0) * 1000


# ── Axelera-backed tracker ────────────────────────────────────────────────────

def build_axelera_tracker(c, raw_store, ort, cfg, ModelBuilder, build_tracker):
    from axelera.runtime import Context

    cfg.defrost()
    cfg.merge_from_file(c['paths']['model_cfg'])
    cfg.CUDA = False
    cfg.freeze()

    model = ModelBuilder().eval()
    ckpt  = torch.load(c['paths']['checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt.get('model', ckpt)))
    tracker = build_tracker(model)

    # Load Axelera compiled models
    ctx        = Context()
    tmpl_model = ctx.load_model(c['paths']['template_encoder_ax'])
    srch_model = ctx.load_model(c['paths']['search_encoder_ax'])

    tmpl_enc_ax = AxeleraEncoder(ctx, tmpl_model, aipu_cores=4)
    srch_enc_ax = AxeleraEncoder(ctx, srch_model, aipu_cores=4)

    # xcorr_head stays on CPU ONNX (dynamic-weight grouped conv not Metis-compilable)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    xcorr_hd = ort.InferenceSession(c['paths']['xcorr_head'], opts,
                                    providers=['CPUExecutionProvider'])
    print('Providers — template/search: Axelera Metis  |  xcorr_head: CPU')

    _zf = {}

    def axelera_template(self, z):
        """[Metis] Encode template patch."""
        z_np = z.cpu().numpy().astype(np.float32)
        feats, t_ms = tmpl_enc_ax.run(z_np)
        if feats[0].shape[-1] > 7:
            c7 = (feats[0].shape[-1] - 7) // 2
            feats = [f[:, :, c7:c7+7, c7:c7+7] for f in feats]
        _zf['feats'] = feats
        print(f'  template_encoder (Metis): {t_ms:.1f} ms  zf={feats[0].shape}')

    def axelera_track(self, x):
        """[Metis] search_encoder  +  [CPU] xcorr_head."""
        x_np = x.cpu().numpy().astype(np.float32)

        # [Metis] search backbone + neck
        xf, _ = srch_enc_ax.run(x_np)

        # [CPU] xcorr + RPN head
        zf = _zf['feats']
        cls_np, loc_np = xcorr_hd.run(None, {
            'zf_0': zf[0], 'zf_1': zf[1], 'zf_2': zf[2],
            'xf_0': xf[0], 'xf_1': xf[1], 'xf_2': xf[2],
        })

        raw_store['cls']         = cls_np
        raw_store['search_crop'] = x_np
        return {'cls': torch.from_numpy(cls_np), 'loc': torch.from_numpy(loc_np)}

    tracker.model.template = types.MethodType(axelera_template, tracker.model)
    tracker.model.track    = types.MethodType(axelera_track,    tracker.model)
    return tracker, tmpl_enc_ax, _zf


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    ap = argparse.ArgumentParser()
    ap.add_argument('--config',     default='configs/track_config_axelera.json')
    ap.add_argument('--output',     default=None)
    ap.add_argument('--max-frames', default=None, type=int)
    ap.add_argument('--pf',         default=None, choices=['true', 'false'])
    args = ap.parse_args()

    import onnxruntime as ort
    from pysot.core.config import cfg
    from pysot.models.model_builder import ModelBuilder
    from pysot.tracker.tracker_builder import build_tracker

    c = load_config(args.config)
    if args.output:     c['paths']['output'] = args.output
    if args.max_frames: c['max_frames'] = args.max_frames
    if args.pf:         c['use_particle_filter'] = (args.pf == 'true')

    pf     = c['particle_filter']
    tu     = c['template_update']
    use_pf = c.get('use_particle_filter', True)

    raw_store = {}
    tracker, _tmpl_enc_ax, _zf = build_axelera_tracker(
        c, raw_store, ort, cfg, ModelBuilder, build_tracker)

    cap_orig  = cv2.VideoCapture(c['paths']['video'])
    sam2_path = c['paths'].get('sam2_video')
    cap_sam   = cv2.VideoCapture(sam2_path) if sam2_path and os.path.exists(sam2_path) else None

    total      = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    fps        = cap_orig.get(cv2.CAP_PROP_FPS) or 25.0
    max_frames = c.get('max_frames') or total
    INIT_BBOX  = c['init_bbox']
    INFO_H     = 80

    _, _f = cap_orig.read(); cap_orig.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if cap_sam: cap_sam.read(); cap_sam.set(cv2.CAP_PROP_POS_FRAMES, 0)
    TARGET_H = _f.shape[0]
    PANEL_W  = _f.shape[1]
    OUT_W    = PANEL_W * 4
    OUT_H    = TARGET_H + 20 + INFO_H

    os.makedirs(os.path.dirname(os.path.abspath(c['paths']['output'])), exist_ok=True)

    mode_str = 'SiamRPN++ Axelera+CPU' + (' + PF' if use_pf else '')
    print(f'Mode: {mode_str}')
    print(f'Output: {OUT_W}x{OUT_H}  {max_frames}/{total} frames @ {fps:.1f}fps')

    _ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUT_W}x{OUT_H}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', 'pipe:',
        '-vcodec', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p',
        c['paths']['output']
    ]
    writer = subprocess.Popen(_ffmpeg_cmd, stdin=subprocess.PIPE)

    N         = pf['n_particles']
    particles = np.zeros((N, 6), np.float32)
    weights   = np.ones(N, np.float32) / N

    cx_out = cy_out = w_out = h_out = 0.0
    cx_search = cy_search = s_x = sx_display = 0.0
    fg = np.zeros(3125)
    score_map   = np.zeros((25, 25))
    track_score = 1.0
    psr = ess   = 0.0
    search_bgr  = np.zeros((255, 255, 3), np.uint8)
    w0 = h0     = 0.0
    initialized = False

    t_enc_total = 0.0
    t_start = time.perf_counter()
    frames_processed = 0

    for fi in range(max_frames):
        ok_o, frame_orig = cap_orig.read()
        if not ok_o:
            break
        if cap_sam:
            ok_s, frame_sam = cap_sam.read()
            if not ok_s:
                frame_sam = frame_orig.copy()
        else:
            frame_sam = frame_orig.copy()

        ih, iw = frame_orig.shape[:2]

        # ── Init ──────────────────────────────────────────────────────────────
        if not initialized:
            cfg.defrost(); cfg.merge_from_file(c['paths']['model_cfg']); cfg.CUDA = False; cfg.freeze()
            tracker.init(frame_orig, INIT_BBOX)
            cx0, cy0 = INIT_BBOX[0] + INIT_BBOX[2] / 2.0, INIT_BBOX[1] + INIT_BBOX[3] / 2.0
            w0, h0   = float(INIT_BBOX[2]), float(INIT_BBOX[3])
            if use_pf:
                particles[:, 0] = cx0 + np.random.randn(N).astype(np.float32) * 2
                particles[:, 1] = cy0 + np.random.randn(N).astype(np.float32) * 2
                particles[:, 2:4] = 0.0
                particles[:, 4] = w0
                particles[:, 5] = h0
                weights[:] = 1.0 / N
            cx_out, cy_out = cx0, cy0
            w_out, h_out   = w0, h0
            cx_search, cy_search = cx0, cy0
            sx_display = 127.0
            initialized = True

        # ── Track ─────────────────────────────────────────────────────────────
        else:
            cx_search = float(tracker.center_pos[0])
            cy_search = float(tracker.center_pos[1])
            tw, th = float(tracker.size[0]), float(tracker.size[1])
            w_z = tw + cfg.TRACK.CONTEXT_AMOUNT * (tw + th)
            h_z = th + cfg.TRACK.CONTEXT_AMOUNT * (tw + th)
            s_x = float(np.sqrt(w_z * h_z)) * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
            sx_display = s_x

            cfg.defrost(); cfg.merge_from_file(c['paths']['model_cfg']); cfg.CUDA = False; cfg.freeze()
            t0_enc = time.perf_counter()
            out = tracker.track(frame_orig)
            t_enc_total += time.perf_counter() - t0_enc
            track_score = float(out.get('best_score', 0))

            fg, score_map = cls_to_scoremap(raw_store['cls'])
            search_bgr    = decode_search(raw_store['search_crop'])
            psr = float((fg.max() - fg.mean()) / (fg.std() + 1e-6))

            if use_pf:
                particles[:, 0] += particles[:, 2] + np.random.randn(N).astype(np.float32) * pf['sigma_pos']
                particles[:, 1] += particles[:, 3] + np.random.randn(N).astype(np.float32) * pf['sigma_pos']
                particles[:, 2] += np.random.randn(N).astype(np.float32) * pf['sigma_vel']
                particles[:, 3] += np.random.randn(N).astype(np.float32) * pf['sigma_vel']
                particles[:, 2]  = np.clip(particles[:, 2], -pf['max_vel'], pf['max_vel'])
                particles[:, 3]  = np.clip(particles[:, 3], -pf['max_vel'], pf['max_vel'])
                particles[:, 4] *= np.exp(np.random.randn(N).astype(np.float32) * pf['sigma_scale'])
                particles[:, 5] *= np.exp(np.random.randn(N).astype(np.float32) * pf['sigma_scale'])
                particles[:, 0]  = np.clip(particles[:, 0], 0, iw)
                particles[:, 1]  = np.clip(particles[:, 1], 0, ih)

                if psr >= 2.0:
                    fx_p, fy_p = particle_to_feat(particles[:, 0], particles[:, 1], cx_search, cy_search, s_x)
                    log_w = score_map[fy_p, fx_p] / pf['tau']
                    log_w -= log_w.max()
                    weights = np.exp(log_w).astype(np.float32)
                    weights /= weights.sum() + 1e-12

                ess = float(1.0 / ((weights ** 2).sum() + 1e-12))
                if ess < N / 2.0:
                    idx = systematic_resample(weights)
                    particles = particles[idx].copy()
                    weights   = np.ones(N, np.float32) / N
                    particles[:, 0] += np.random.randn(N).astype(np.float32) * pf['roughen_std']
                    particles[:, 1] += np.random.randn(N).astype(np.float32) * pf['roughen_std']

                cx_out = float(np.dot(weights, particles[:, 0]))
                cy_out = float(np.dot(weights, particles[:, 1]))
                w_out  = float(tracker.size[0])
                h_out  = float(tracker.size[1])
                tracker.center_pos = np.array([cx_out, cy_out])
                ess = float(1.0 / ((weights ** 2).sum() + 1e-12))

            else:
                bbox = out.get('bbox', None)
                if bbox is not None:
                    cx_out = float(bbox[0] + bbox[2] / 2)
                    cy_out = float(bbox[1] + bbox[3] / 2)
                    w_out, h_out = float(bbox[2]), float(bbox[3])
                else:
                    cx_out, cy_out = float(tracker.center_pos[0]), float(tracker.center_pos[1])
                    w_out, h_out   = float(tracker.size[0]), float(tracker.size[1])
                ess = float(N)

            # Template update
            if fi % tu['freq'] == 0 and psr > tu['min_psr'] and w_out < w0 * 3 and h_out < h0 * 3:
                _ctx  = cfg.TRACK.CONTEXT_AMOUNT * (w_out + h_out)
                _sz2  = round(float(np.sqrt((w_out + _ctx) * (h_out + _ctx))))
                _z_new = tracker.get_subwindow(frame_orig, np.array([cx_out, cy_out]),
                                               cfg.TRACK.EXEMPLAR_SIZE, _sz2, tracker.channel_average)
                _z_np = _z_new.cpu().numpy().astype(np.float32)
                _nf, _ = _tmpl_enc_ax.run(_z_np)
                if _nf[0].shape[-1] > 7:
                    _c7 = (_nf[0].shape[-1] - 7) // 2
                    _nf = [f[:, :, _c7:_c7+7, _c7:_c7+7] for f in _nf]
                _zf['feats'] = [(1 - tu['alpha']) * _zf['feats'][i] + tu['alpha'] * _nf[i]
                                for i in range(3)]

        # ── Visualise ─────────────────────────────────────────────────────────
        fvis = frame_orig.copy()
        x1, y1 = int(cx_out - w_out / 2), int(cy_out - h_out / 2)
        cv2.rectangle(fvis, (x1, y1), (x1 + int(w_out), y1 + int(h_out)), (0, 255, 80), 2)
        cv2.circle(fvis, (int(cx_out), int(cy_out)), 3, (0, 255, 255), -1)
        sx1 = int(cx_out - sx_display / 2); sy1 = int(cy_out - sx_display / 2)
        cv2.rectangle(fvis, (sx1, sy1), (sx1 + int(sx_display), sy1 + int(sx_display)), (255, 100, 0), 1)
        if use_pf:
            for pi in np.argsort(weights)[-50:]:
                ppx, ppy = int(particles[pi, 0]), int(particles[pi, 1])
                if 0 <= ppx < iw and 0 <= ppy < ih:
                    cv2.circle(fvis, (ppx, ppy), max(1, int(weights[pi] * N * 0.6)), (0, 200, 255), -1)

        p1 = add_label(fit_panel(fvis, TARGET_H, PANEL_W),
                       f'{"PF+" if use_pf else ""}SiamRPN++ [Axelera]  score={track_score:.3f}  f={fi}')

        sq = TARGET_H
        sm_norm = ((score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8) * 255).astype(np.uint8)
        hmap = cv2.applyColorMap(cv2.resize(sm_norm, (sq, sq), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_JET)
        if fi > 0 and s_x > 0 and use_pf:
            fx_v, fy_v = particle_to_feat(particles[:, 0], particles[:, 1], cx_search, cy_search, s_x)
            cell = sq / 25.0
            for pi in np.argsort(weights)[-100:]:
                svx = int(fx_v[pi] * cell + cell / 2); svy = int(fy_v[pi] * cell + cell / 2)
                if 0 <= svx < sq and 0 <= svy < sq:
                    cv2.circle(hmap, (svx, svy), 3, (255, 255, 255), -1)
        p2 = add_label(fit_panel(hmap, TARGET_H, PANEL_W), f'Score map  PSR={psr:.1f}  ESS={ess:.0f}/{N}')
        p3 = add_label(fit_panel(search_bgr if search_bgr.sum() > 0 else np.zeros((sq, sq, 3), np.uint8),
                                 TARGET_H, PANEL_W), f'Search crop  s_x={sx_display:.0f}px')
        p4 = add_label(fit_panel(frame_sam, TARGET_H, PANEL_W), 'SAM2 / ref')

        PANEL_H = TARGET_H + 20
        top_row = np.hstack([cv2.resize(p, (PANEL_W, PANEL_H)) if p.shape[:2] != (PANEL_H, PANEL_W) else p
                             for p in [p1, p2, p3, p4]])
        if top_row.shape[1] != OUT_W:
            top_row = cv2.resize(top_row, (OUT_W, PANEL_H))

        info = np.zeros((INFO_H, OUT_W, 3), np.uint8)
        vx_m = float(np.dot(weights, particles[:, 2])) if use_pf else 0.0
        vy_m = float(np.dot(weights, particles[:, 3])) if use_pf else 0.0
        put(info, f'Frame: {fi:5d} / {max_frames}',          6,  18, 0.48)
        put(info, f'Track score: {track_score:.4f}',          6,  38, 0.48,
            (0, 255, 80) if track_score > 0.5 else (0, 80, 255))
        put(info, f'PSR={psr:.1f}',                           6,  58, 0.42,
            (0, 230, 100) if psr > tu['min_psr'] else (0, 80, 255))
        put(info, f'cx={cx_out:.1f}  cy={cy_out:.1f}',     260,  18, 0.42)
        put(info, f'w={w_out:.1f}  h={h_out:.1f}',         260,  38, 0.42)
        put(info, f'vel=({vx_m:+.1f},{vy_m:+.1f})px/f',   260,  58, 0.42)
        put(info, f'ESS={ess:.0f}/{N}',                     480,  18, 0.42,
            (0, 200, 100) if ess > N * 0.4 else (0, 80, 255))
        put(info, f"mode={'PF+Axelera' if use_pf else 'Axelera'}", 480, 38, 0.42, (160, 160, 160))
        put(info, '[Metis] tmpl+srch  [CPU] xcorr',          480,  58, 0.42, (100, 200, 100))

        frame_out = np.vstack([top_row, info])
        if frame_out.shape[:2] != (OUT_H, OUT_W):
            frame_out = cv2.resize(frame_out, (OUT_W, OUT_H))
        writer.stdin.write(frame_out.tobytes())

        frames_processed += 1
        if fi % 50 == 0 or fi < 2:
            elapsed = time.perf_counter() - t_start
            fps_now = frames_processed / elapsed if elapsed > 0 else 0.0
            print(f'  frame {fi:5d}/{max_frames}  score={track_score:.3f}'
                  f'  PSR={psr:.1f}  ESS={ess:.0f}'
                  f'  pos=({cx_out:.0f},{cy_out:.0f})'
                  f'  fps={fps_now:.2f}')

    cap_orig.release()
    if cap_sam: cap_sam.release()
    writer.stdin.close()
    writer.wait()

    elapsed = time.perf_counter() - t_start
    actual_fps = frames_processed / elapsed if elapsed > 0 else 0.0
    enc_avg_ms = (t_enc_total / max(frames_processed - 1, 1)) * 1000

    print(f'\nSaved: {c["paths"]["output"]}')
    print(f'Processed {frames_processed} frames in {elapsed:.2f}s  →  {actual_fps:.2f} fps')
    print(f'Avg search_enc (Metis): {enc_avg_ms:.1f} ms/frame')


if __name__ == '__main__':
    main()
