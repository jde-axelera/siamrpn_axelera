#!/usr/bin/env python3
"""
track_split_axelera_nopf.py — SiamRPN++ on Axelera Metis, no particle filter
==============================================================================
Single-panel output: tracked frame + bounding box only.
No score map, no search crop, no SAM2 panel.

Uses the same config file as track_split_axelera.py.

Usage:
    PYTHONPATH=/path/to/pysot \
        python track_split_axelera_nopf.py \
        --config configs/track_config_axelera.json

Output layout (W × 550):
    [label bar 20px]
    [tracked frame W×480 with bbox overlay]
    [info bar 50px: frame#, score, fps, bbox]

A _stats.json file is saved alongside the video with fps, frame count, etc.
"""

import argparse, json, os, sys, time, types, subprocess
import numpy as np
import cv2
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)


def load_config(path):
    with open(path) as f:
        return json.load(f)


def cls_to_scoremap(cls_raw, n_anchors=5, size=25):
    s = cls_raw.transpose(1, 2, 3, 0).reshape(2, -1).T
    e = np.exp(s - s.max(axis=1, keepdims=True))
    fg = (e / e.sum(axis=1, keepdims=True))[:, 1]
    return fg


def put(img, txt, x, y, scale=0.45, col=(200, 200, 200)):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


# ── Axelera inference wrappers (same as track_split_axelera.py) ───────────────

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
            img = img.transpose(0, 2, 3, 1).squeeze(0)
        elif img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
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

    ctx        = Context()
    tmpl_model = ctx.load_model(c['paths']['template_encoder_ax'])
    srch_model = ctx.load_model(c['paths']['search_encoder_ax'])

    tmpl_enc_ax = AxeleraEncoder(ctx, tmpl_model, aipu_cores=4)
    srch_enc_ax = AxeleraEncoder(ctx, srch_model, aipu_cores=4)

    import platform as _platform
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    _xnnpack_opts = {'intra_op_num_threads': 4} if _platform.machine() == 'aarch64' else {}
    _cpu_providers = (
        [('XNNPACKExecutionProvider', _xnnpack_opts), 'CPUExecutionProvider']
        if _platform.machine() == 'aarch64'
        else ['CPUExecutionProvider']
    )
    xcorr_hd = ort.InferenceSession(c['paths']['xcorr_head'], opts, providers=_cpu_providers)
    _xcorr_ep = 'XNNPACK+CPU' if _platform.machine() == 'aarch64' else 'CPU'
    print(f'Providers — template/search: Axelera Metis  |  xcorr_head: {_xcorr_ep}')
    if _platform.machine() == 'aarch64' and not os.environ.get('AXELERA_CONFIGURE_BOARD'):
        print('  [RPi5] Tip: if inference crashes, set AXELERA_CONFIGURE_BOARD=,20')

    _zf = {}

    def axelera_template(self, z):
        z_np = z.cpu().numpy().astype(np.float32)
        feats, t_ms = tmpl_enc_ax.run(z_np)
        if feats[0].shape[-1] > 7:
            c7 = (feats[0].shape[-1] - 7) // 2
            feats = [f[:, :, c7:c7+7, c7:c7+7] for f in feats]
        _zf['feats'] = feats
        print(f'  template_encoder (Metis): {t_ms:.1f} ms  zf={feats[0].shape}')

    def axelera_track(self, x):
        x_np = x.cpu().numpy().astype(np.float32)
        xf, _ = srch_enc_ax.run(x_np)
        zf = _zf['feats']
        cls_np, loc_np = xcorr_hd.run(None, {
            'zf_0': zf[0], 'zf_1': zf[1], 'zf_2': zf[2],
            'xf_0': xf[0], 'xf_1': xf[1], 'xf_2': xf[2],
        })
        raw_store['cls'] = cls_np
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
    args = ap.parse_args()

    import onnxruntime as ort
    from pysot.core.config import cfg
    from pysot.models.model_builder import ModelBuilder
    from pysot.tracker.tracker_builder import build_tracker

    c = load_config(args.config)
    if args.output:
        c['paths']['output'] = args.output
    if args.max_frames:
        c['max_frames'] = args.max_frames

    tu = c.get('template_update', {'freq': 30, 'alpha': 0.25, 'min_psr': 6.0})

    raw_store = {}
    tracker, _tmpl_enc_ax, _zf = build_axelera_tracker(
        c, raw_store, ort, cfg, ModelBuilder, build_tracker)

    cap       = cv2.VideoCapture(c['paths']['video'])
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    max_frames = c.get('max_frames') or total
    INIT_BBOX  = c['init_bbox']

    _, _f = cap.read(); cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    H, W  = _f.shape[:2]
    LBL_H = 20
    INF_H = 50
    OUT_H = LBL_H + H + INF_H
    OUT_W = W

    out_path = c['paths']['output']
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUT_W}x{OUT_H}', '-pix_fmt', 'bgr24', '-r', str(src_fps),
        '-i', 'pipe:',
        '-vcodec', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p',
        out_path
    ]
    writer = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    # Score below this → fall back to velocity-predicted position and restore
    # tracker's internal state so the next search stays on-target.
    SCORE_GATE     = 0.3
    VEL_EMA_ALPHA  = 0.3   # weight of newest delta in velocity EMA
    VEL_DAMPEN     = 0.8   # per-frame velocity damping while score is low
                           # (effect decays to ~10% after 10 frames → ~30px max drift)

    cx_out = cy_out = w_out = h_out = 0.0
    w0 = h0 = 0.0
    last_good_cx = last_good_cy = 0.0
    last_good_w  = last_good_h  = 0.0
    vx = vy = 0.0          # EMA velocity estimate (pixels / frame)
    track_score = 1.0
    psr = 0.0
    initialized = False

    t_start = time.perf_counter()
    frames_processed = 0
    t_enc_total = 0.0

    for fi in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break

        if not initialized:
            cfg.defrost(); cfg.merge_from_file(c['paths']['model_cfg']); cfg.CUDA = False; cfg.freeze()
            tracker.init(frame, INIT_BBOX)
            cx_out = last_good_cx = INIT_BBOX[0] + INIT_BBOX[2] / 2.0
            cy_out = last_good_cy = INIT_BBOX[1] + INIT_BBOX[3] / 2.0
            w_out = last_good_w = w0 = float(INIT_BBOX[2])
            h_out = last_good_h = h0 = float(INIT_BBOX[3])
            initialized = True
        else:
            cfg.defrost(); cfg.merge_from_file(c['paths']['model_cfg']); cfg.CUDA = False; cfg.freeze()
            t0_enc = time.perf_counter()
            out = tracker.track(frame)
            t_enc_total += time.perf_counter() - t0_enc
            track_score = float(out.get('best_score', 0))

            bbox = out.get('bbox', None)
            if track_score >= SCORE_GATE:
                if bbox is not None:
                    new_cx = float(bbox[0] + bbox[2] / 2)
                    new_cy = float(bbox[1] + bbox[3] / 2)
                    w_out, h_out = float(bbox[2]), float(bbox[3])
                else:
                    new_cx = float(tracker.center_pos[0])
                    new_cy = float(tracker.center_pos[1])
                    w_out, h_out = float(tracker.size[0]), float(tracker.size[1])
                # Update velocity EMA from the detection delta
                vx = (1 - VEL_EMA_ALPHA) * vx + VEL_EMA_ALPHA * (new_cx - last_good_cx)
                vy = (1 - VEL_EMA_ALPHA) * vy + VEL_EMA_ALPHA * (new_cy - last_good_cy)
                cx_out, cy_out = new_cx, new_cy
                last_good_cx, last_good_cy = cx_out, cy_out
                last_good_w,  last_good_h  = w_out, h_out
            else:
                # Low confidence: advance by dampened velocity so the search
                # window tracks where the target is likely going rather than
                # stalling at a stale position indefinitely.
                vx *= VEL_DAMPEN
                vy *= VEL_DAMPEN
                last_good_cx = np.clip(last_good_cx + vx, 0, frame.shape[1])
                last_good_cy = np.clip(last_good_cy + vy, 0, frame.shape[0])
                cx_out, cy_out = last_good_cx, last_good_cy
                w_out,  h_out  = last_good_w,  last_good_h
                tracker.size   = np.array([last_good_w, last_good_h])

            # Always keep tracker searching from our best position estimate.
            tracker.center_pos = np.array([cx_out, cy_out])

            cls_raw = raw_store.get('cls')
            if cls_raw is not None:
                fg = cls_to_scoremap(cls_raw)
                psr = float((fg.max() - fg.mean()) / (fg.std() + 1e-6))

            if fi % tu['freq'] == 0 and psr > tu['min_psr']:
                _ctx = cfg.TRACK.CONTEXT_AMOUNT * (w_out + h_out)
                _sz2 = round(float(np.sqrt((w_out + _ctx) * (h_out + _ctx))))
                _z_new = tracker.get_subwindow(frame, np.array([cx_out, cy_out]),
                                               cfg.TRACK.EXEMPLAR_SIZE, _sz2, tracker.channel_average)
                _z_np = _z_new.cpu().numpy().astype(np.float32)
                _nf, _ = _tmpl_enc_ax.run(_z_np)
                if _nf[0].shape[-1] > 7:
                    _c7 = (_nf[0].shape[-1] - 7) // 2
                    _nf = [f[:, :, _c7:_c7+7, _c7:_c7+7] for f in _nf]
                _zf['feats'] = [(1 - tu['alpha']) * _zf['feats'][i] + tu['alpha'] * _nf[i]
                                for i in range(3)]

        # ── Draw bbox ────────────────────────────────────────────────────────
        fvis = frame.copy()
        x1 = int(cx_out - w_out / 2)
        y1 = int(cy_out - h_out / 2)
        cv2.rectangle(fvis, (x1, y1), (x1 + int(w_out), y1 + int(h_out)), (0, 255, 80), 2)
        cv2.circle(fvis, (int(cx_out), int(cy_out)), 3, (0, 255, 255), -1)

        elapsed  = time.perf_counter() - t_start
        fps_live = frames_processed / elapsed if elapsed > 0 else 0.0

        lbl = np.full((LBL_H, OUT_W, 3), (15, 15, 15), dtype=np.uint8)
        cv2.putText(lbl, f'SiamRPN++ [Axelera, no PF]  f={fi}', (4, LBL_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)

        info = np.zeros((INF_H, OUT_W, 3), np.uint8)
        put(info, f'Frame: {fi:5d} / {max_frames}',                  6,  18, 0.45)
        put(info, f'Score: {track_score:.4f}',                        6,  38, 0.45,
            (0, 255, 80) if track_score > 0.5 else (0, 80, 255))
        put(info, f'FPS: {fps_live:.2f}',                           260,  18, 0.45)
        put(info, f'cx={cx_out:.1f}  cy={cy_out:.1f}  w={w_out:.1f}  h={h_out:.1f}',
                                                                    260,  38, 0.45)

        frame_out = np.vstack([lbl, fvis, info])
        writer.stdin.write(frame_out.tobytes())

        frames_processed += 1
        print(f'  frame {fi:5d}/{max_frames}  score={track_score:.4f}'
              f'  PSR={psr:.2f}  cx={cx_out:.1f}  cy={cy_out:.1f}'
              f'  w={w_out:.1f}  h={h_out:.1f}  fps={fps_live:.2f}')

    cap.release()
    writer.stdin.close()
    writer.wait()

    elapsed     = time.perf_counter() - t_start
    actual_fps  = frames_processed / elapsed if elapsed > 0 else 0.0
    enc_avg_ms  = (t_enc_total / max(frames_processed - 1, 1)) * 1000

    print(f'\nSaved: {out_path}')
    print(f'Processed {frames_processed} frames in {elapsed:.2f}s  →  {actual_fps:.2f} fps')
    print(f'Avg inference (Metis + CPU xcorr): {enc_avg_ms:.1f} ms/frame')

    stats_path = os.path.splitext(out_path)[0] + '_stats.json'
    with open(stats_path, 'w') as f:
        json.dump({
            'fps':        actual_fps,
            'frames':     frames_processed,
            'elapsed_s':  elapsed,
            'enc_avg_ms': enc_avg_ms,
            'mode':       'axelera_nopf',
        }, f, indent=2)
    print(f'Stats: {stats_path}')


if __name__ == '__main__':
    main()
