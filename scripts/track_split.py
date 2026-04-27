#!/usr/bin/env python3
"""
track_split.py — SiamRPN++ tracker using three separate ONNX models
====================================================================

Inference pipeline per frame:
  [Metis] template_encoder.onnx  (1×, at init / template update)
  [Metis] search_encoder.onnx    (every frame)
  [CPU]   xcorr_head.onnx        (every frame — dynamic-weight grouped conv)
  [CPU]   post-processing + optional particle filter

Usage:
    conda activate pysot
    python scripts/track_split.py --config configs/track_config.json

Config:
    paths.template_encoder  — backbone+neck for template patch
    paths.search_encoder    — backbone+neck for search patch   (Metis target)
    paths.xcorr_head        — RPN head (xcorr + cls/loc heads) (CPU)
    use_particle_filter     — true/false
    max_frames              — int or null (full video)
"""

import argparse, json, os, sys, time, types, subprocess
import numpy as np
import cv2
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..', '..', 'pysot'))


def load_config(path):
    with open(path) as f:
        return json.load(f)


# ── Constants ─────────────────────────────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)


# ── Helpers ───────────────────────────────────────────────────────────────────

def systematic_resample(weights):
    N = len(weights)
    weights = weights / (weights.sum() + 1e-300)  # guard against fp sum != 1
    positions = (np.random.random() + np.arange(N)) / N
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # force exact 1.0 to prevent infinite loop at boundary
    i, j = 0, 0
    indices = np.zeros(N, dtype=int)
    while i < N:
        if positions[i] < cumsum[j]:
            indices[i] = j
            i += 1
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


# ── ONNX-backed tracker (three-model split) ───────────────────────────────────

def build_split_tracker(c, raw_store, ort, cfg, ModelBuilder, build_tracker):
    cfg.defrost()
    cfg.merge_from_file(c['paths']['model_cfg'])
    cfg.CUDA = False
    cfg.freeze()

    model = ModelBuilder().eval()
    ckpt = torch.load(c['paths']['checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt.get('model', ckpt)))
    tracker = build_tracker(model)

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    gpu_providers  = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
        if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    cpu_providers  = ['CPUExecutionProvider']
    print(f'ONNX providers — Metis targets: {gpu_providers}  |  xcorr_head: {cpu_providers}')

    tmpl_enc  = ort.InferenceSession(c['paths']['template_encoder'], opts, providers=gpu_providers)
    srch_enc  = ort.InferenceSession(c['paths']['search_encoder'],   opts, providers=gpu_providers)
    xcorr_hd  = ort.InferenceSession(c['paths']['xcorr_head'],       opts, providers=cpu_providers)

    _zf = {}

    def onnx_template(self, z):
        """[Metis] Encode template patch → zf features (run once)."""
        _zf['feats'] = tmpl_enc.run(None, {'template': z.cpu().numpy().astype(np.float32)})

    def onnx_track(self, x):
        """[Metis] search_encoder + [CPU] xcorr_head."""
        x_np = x.cpu().numpy().astype(np.float32)

        # [Metis] search backbone + neck
        xf = srch_enc.run(None, {'search': x_np})

        # [CPU] xcorr + RPN head
        zf = _zf['feats']
        cls_np, loc_np = xcorr_hd.run(None, {
            'zf_0': zf[0], 'zf_1': zf[1], 'zf_2': zf[2],
            'xf_0': xf[0], 'xf_1': xf[1], 'xf_2': xf[2],
        })

        raw_store['cls']         = cls_np
        raw_store['search_crop'] = x_np
        return {'cls': torch.from_numpy(cls_np), 'loc': torch.from_numpy(loc_np)}

    tracker.model.template = types.MethodType(onnx_template, tracker.model)
    tracker.model.track    = types.MethodType(onnx_track,    tracker.model)
    return tracker, tmpl_enc, _zf


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    ap = argparse.ArgumentParser()
    ap.add_argument('--config',     default='configs/track_config.json')
    ap.add_argument('--output',     default=None,  help='override output path')
    ap.add_argument('--max-frames', default=None,  type=int, help='override max_frames')
    ap.add_argument('--pf',         default=None,  choices=['true','false'], help='override use_particle_filter')
    args = ap.parse_args()

    import onnxruntime as ort
    from pysot.core.config import cfg
    from pysot.models.model_builder import ModelBuilder
    from pysot.tracker.tracker_builder import build_tracker

    c      = load_config(args.config)
    if args.output:
        c['paths']['output'] = args.output
    if args.max_frames is not None:
        c['max_frames'] = args.max_frames
    if args.pf is not None:
        c['use_particle_filter'] = (args.pf == 'true')

    pf     = c['particle_filter']
    tu     = c['template_update']
    use_pf = c.get('use_particle_filter', True)

    raw_store = {}
    tracker, _tmpl_enc, _zf = build_split_tracker(c, raw_store, ort, cfg, ModelBuilder, build_tracker)

    cap_orig   = cv2.VideoCapture(c['paths']['video'])
    cap_sam    = cv2.VideoCapture(c['paths']['sam2_video'])
    total      = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    fps        = cap_orig.get(cv2.CAP_PROP_FPS) or 25.0
    max_frames = c['max_frames'] or total
    INIT_BBOX  = c['init_bbox']
    INFO_H     = 80

    _, _f = cap_orig.read(); cap_orig.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_sam.read();           cap_sam.set(cv2.CAP_PROP_POS_FRAMES, 0)
    TARGET_H = _f.shape[0]
    PANEL_W  = _f.shape[1]
    OUT_W    = PANEL_W * 4
    OUT_H    = TARGET_H + 20 + INFO_H

    mode_str = 'SiamRPN++ split (Metis+CPU)' + (' + PF' if use_pf else '')
    print(f'Mode: {mode_str}')
    print(f'Output: {OUT_W}x{OUT_H}  {max_frames}/{total} frames @ {fps:.1f}fps')
    print(f'  [Metis] template_encoder + search_encoder')
    print(f'  [CPU]   xcorr_head + post-proc' + (' + PF' if use_pf else ''))

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

    t_start = time.perf_counter()
    frames_processed = 0

    for fi in range(max_frames):
        ok_o, frame_orig = cap_orig.read()
        ok_s, frame_sam  = cap_sam.read()
        if not (ok_o and ok_s):
            break

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
            out = tracker.track(frame_orig)
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
                    w_out  = float(bbox[2])
                    h_out  = float(bbox[3])
                else:
                    cx_out = float(tracker.center_pos[0])
                    cy_out = float(tracker.center_pos[1])
                    w_out  = float(tracker.size[0])
                    h_out  = float(tracker.size[1])
                ess = float(N)

            # Template update (both modes)
            if fi % tu['freq'] == 0 and psr > tu['min_psr'] and w_out < w0 * 3 and h_out < h0 * 3:
                _ctx  = cfg.TRACK.CONTEXT_AMOUNT * (w_out + h_out)
                _sz2  = round(float(np.sqrt((w_out + _ctx) * (h_out + _ctx))))
                _z_new = tracker.get_subwindow(frame_orig, np.array([cx_out, cy_out]),
                                               cfg.TRACK.EXEMPLAR_SIZE, _sz2, tracker.channel_average)
                _nf = _tmpl_enc.run(None, {'template': _z_new.cpu().numpy().astype(np.float32)})
                _zf['feats'] = [(1 - tu['alpha']) * _zf['feats'][i] + tu['alpha'] * _nf[i] for i in range(3)]

        # ── Visualise ─────────────────────────────────────────────────────────
        fvis = frame_orig.copy()
        x1, y1 = int(cx_out - w_out / 2), int(cy_out - h_out / 2)
        cv2.rectangle(fvis, (x1, y1), (x1 + int(w_out), y1 + int(h_out)), (0, 255, 80), 2)
        cv2.circle(fvis, (int(cx_out), int(cy_out)), 3, (0, 255, 255), -1)
        sx1, sy1 = int(cx_out - sx_display / 2), int(cy_out - sx_display / 2)
        cv2.rectangle(fvis, (sx1, sy1), (sx1 + int(sx_display), sy1 + int(sx_display)), (255, 100, 0), 1)
        if use_pf:
            for pi in np.argsort(weights)[-50:]:
                ppx, ppy = int(particles[pi, 0]), int(particles[pi, 1])
                if 0 <= ppx < iw and 0 <= ppy < ih:
                    cv2.circle(fvis, (ppx, ppy), max(1, int(weights[pi] * N * 0.6)), (0, 200, 255), -1)
        panel1_label = f'{"PF+" if use_pf else ""}SiamRPN++ [split]  score={track_score:.3f}  f={fi}'
        p1 = add_label(fit_panel(fvis, TARGET_H, PANEL_W), panel1_label)

        sq = TARGET_H
        sm_norm = ((score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8) * 255).astype(np.uint8)
        hmap = cv2.applyColorMap(cv2.resize(sm_norm, (sq, sq), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_JET)
        if fi > 0 and s_x > 0 and use_pf:
            fx_v, fy_v = particle_to_feat(particles[:, 0], particles[:, 1], cx_search, cy_search, s_x)
            cell = sq / 25.0
            for pi in np.argsort(weights)[-100:]:
                svx = int(fx_v[pi] * cell + cell / 2)
                svy = int(fy_v[pi] * cell + cell / 2)
                if 0 <= svx < sq and 0 <= svy < sq:
                    cv2.circle(hmap, (svx, svy), 3, (255, 255, 255), -1)
        p2 = add_label(fit_panel(hmap, TARGET_H, PANEL_W), f'Score map  PSR={psr:.1f}  ESS={ess:.0f}/{N}')
        p3 = add_label(fit_panel(search_bgr if search_bgr.sum() > 0 else np.zeros((sq, sq, 3), np.uint8),
                                 TARGET_H, PANEL_W), f'Search crop  s_x={sx_display:.0f}px')
        p4 = add_label(fit_panel(cv2.rotate(frame_sam, cv2.ROTATE_90_CLOCKWISE), TARGET_H, PANEL_W), 'SAM2')

        PANEL_H = TARGET_H + 20
        top_row = np.hstack([cv2.resize(p, (PANEL_W, PANEL_H)) if p.shape[:2] != (PANEL_H, PANEL_W) else p
                             for p in [p1, p2, p3, p4]])
        if top_row.shape[1] != OUT_W:
            top_row = cv2.resize(top_row, (OUT_W, PANEL_H))

        info = np.zeros((INFO_H, OUT_W, 3), np.uint8)
        vx_m = float(np.dot(weights, particles[:, 2])) if use_pf else 0.0
        vy_m = float(np.dot(weights, particles[:, 3])) if use_pf else 0.0
        put(info, f'Frame: {fi:5d} / {max_frames}',            6,  18, 0.48)
        put(info, f'Track score: {track_score:.4f}',            6,  38, 0.48, (0, 255, 80) if track_score > 0.5 else (0, 80, 255))
        put(info, f'PSR={psr:.1f}',                             6,  58, 0.42, (0, 230, 100) if psr > tu['min_psr'] else (0, 80, 255))
        put(info, f'cx={cx_out:.1f}  cy={cy_out:.1f}',       260,  18, 0.42)
        put(info, f'w={w_out:.1f}  h={h_out:.1f}',           260,  38, 0.42)
        put(info, f'vel=({vx_m:+.1f},{vy_m:+.1f})px/f',     260,  58, 0.42)
        put(info, f'ESS={ess:.0f}/{N}',                       480,  18, 0.42, (0, 200, 100) if ess > N * 0.4 else (0, 80, 255))
        put(info, f"mode={'PF+split' if use_pf else 'split'}", 480, 38, 0.42, (160, 160, 160))
        put(info, f"[Metis] tmpl_enc + srch_enc",              480, 58, 0.42, (100, 200, 100))

        frame_out = np.vstack([top_row, info])
        if frame_out.shape[:2] != (OUT_H, OUT_W):
            frame_out = cv2.resize(frame_out, (OUT_W, OUT_H))
        writer.stdin.write(frame_out.tobytes())

        frames_processed += 1
        if fi % 500 == 0 or fi < 2:
            print(f'  frame {fi:5d}/{max_frames}  score={track_score:.3f}'
                  f'  PSR={psr:.1f}  ESS={ess:.0f}'
                  f'  pos=({cx_out:.0f},{cy_out:.0f})'
                  f'  vel=({vx_m:+.1f},{vy_m:+.1f})')

    cap_orig.release()
    cap_sam.release()
    writer.stdin.close()
    writer.wait()
    elapsed = time.perf_counter() - t_start
    actual_fps = frames_processed / elapsed if elapsed > 0 else 0.0
    print(f'\nSaved: {c["paths"]["output"]}')
    print(f'Processed {frames_processed} frames in {elapsed:.2f}s  →  {actual_fps:.2f} fps (wall clock)')


if __name__ == '__main__':
    main()
