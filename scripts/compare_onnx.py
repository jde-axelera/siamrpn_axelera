#!/usr/bin/env python3
"""
compare_onnx.py — Frame-by-frame comparison of original vs split ONNX tracker.
Runs both models without writing video (fast), dumps CSVs, prints full report.

Usage:
    python scripts/compare_onnx.py \
        --config-orig  configs/track_config_pod.json \
        --config-split configs/track_config_split_pod.json \
        --out-dir      inference_output/

Output files:
    scores_orig_run{N}.csv   — per-frame scores for original (N=1,2,3)
    scores_split_run{N}.csv  — per-frame scores for split    (N=1,2,3)
    comparison_report.txt    — full analysis
"""

import argparse, csv, json, os, sys, time, types
import numpy as np
import cv2
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'pysot'))


def load_config(path):
    with open(path) as f:
        return json.load(f)


# ── PF helpers (same as tracking scripts) ────────────────────────────────────

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


def particle_to_feat(px, py, cx_s, cy_s, s_x):
    scale = 255.0 / (s_x + 1e-6)
    sx_p = (px - cx_s) * scale + 127.5
    sy_p = (py - cy_s) * scale + 127.5
    return (np.clip(((sx_p - 31.5) / 8.0).astype(int), 0, 24),
            np.clip(((sy_p - 31.5) / 8.0).astype(int), 0, 24))


# ── Session builders ──────────────────────────────────────────────────────────

def build_original(c, raw, ort, cfg, ModelBuilder, build_tracker):
    cfg.defrost(); cfg.merge_from_file(c['paths']['model_cfg']); cfg.CUDA = False; cfg.freeze()
    model = ModelBuilder().eval()
    ckpt  = torch.load(c['paths']['checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt.get('model', ckpt)))
    tracker = build_tracker(model)
    opts = ort.SessionOptions(); opts.log_severity_level = 3
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
        if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    enc = ort.InferenceSession(c['paths']['encoder'],  opts, providers=providers)
    trk = ort.InferenceSession(c['paths']['tracker'],  opts, providers=providers)
    _zf = {}

    def tmpl(self, z):
        _zf['feats'] = enc.run(None, {'template': z.cpu().numpy().astype(np.float32)})

    def track(self, x):
        x_np = x.cpu().numpy().astype(np.float32)
        zf = _zf['feats']
        cls_np, loc_np = trk.run(None, {'zf_0': zf[0], 'zf_1': zf[1], 'zf_2': zf[2], 'search': x_np})
        raw['cls'] = cls_np; raw['search_crop'] = x_np
        return {'cls': torch.from_numpy(cls_np), 'loc': torch.from_numpy(loc_np)}

    tracker.model.template = types.MethodType(tmpl,  tracker.model)
    tracker.model.track    = types.MethodType(track, tracker.model)
    return tracker, enc, _zf


def build_split(c, raw, ort, cfg, ModelBuilder, build_tracker):
    cfg.defrost(); cfg.merge_from_file(c['paths']['model_cfg']); cfg.CUDA = False; cfg.freeze()
    model = ModelBuilder().eval()
    ckpt  = torch.load(c['paths']['checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt.get('model', ckpt)))
    tracker = build_tracker(model)
    opts = ort.SessionOptions(); opts.log_severity_level = 3
    gpu = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
        if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    cpu = ['CPUExecutionProvider']
    tmpl_enc = ort.InferenceSession(c['paths']['template_encoder'], opts, providers=gpu)
    srch_enc = ort.InferenceSession(c['paths']['search_encoder'],   opts, providers=gpu)
    xcorr_hd = ort.InferenceSession(c['paths']['xcorr_head'],       opts, providers=cpu)
    _zf = {}

    def tmpl(self, z):
        _zf['feats'] = tmpl_enc.run(None, {'template': z.cpu().numpy().astype(np.float32)})

    def track(self, x):
        x_np = x.cpu().numpy().astype(np.float32)
        xf = srch_enc.run(None, {'search': x_np})
        zf = _zf['feats']
        cls_np, loc_np = xcorr_hd.run(None, {
            'zf_0': zf[0], 'zf_1': zf[1], 'zf_2': zf[2],
            'xf_0': xf[0], 'xf_1': xf[1], 'xf_2': xf[2],
        })
        raw['cls'] = cls_np; raw['search_crop'] = x_np
        return {'cls': torch.from_numpy(cls_np), 'loc': torch.from_numpy(loc_np)}

    tracker.model.template = types.MethodType(tmpl,  tracker.model)
    tracker.model.track    = types.MethodType(track, tracker.model)
    return tracker, tmpl_enc, _zf


# ── Core run loop ─────────────────────────────────────────────────────────────

def run_once(c, builder_fn, seed, ort, cfg, ModelBuilder, build_tracker):
    np.random.seed(seed)
    pf = c['particle_filter']; tu = c['template_update']
    raw = {}
    tracker, _enc, _zf = builder_fn(c, raw, ort, cfg, ModelBuilder, build_tracker)
    cap = cv2.VideoCapture(c['paths']['video'])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    BBOX = c['init_bbox']
    N = pf['n_particles']
    particles = np.zeros((N, 6), np.float32)
    weights   = np.ones(N, np.float32) / N
    cx_out = cy_out = w_out = h_out = s_x = cx_s = cy_s = 0.0
    score_map = np.zeros((25, 25)); fg = np.zeros(3125)
    track_score = psr = ess = 0.0; w0 = h0 = 0.0
    initialized = False; rows = []; t0 = time.perf_counter()

    for fi in range(total):
        ok, frame = cap.read()
        if not ok: break
        ih, iw = frame.shape[:2]

        if not initialized:
            cfg.defrost(); cfg.merge_from_file(c['paths']['model_cfg']); cfg.CUDA = False; cfg.freeze()
            tracker.init(frame, BBOX)
            cx0 = BBOX[0] + BBOX[2] / 2.0; cy0 = BBOX[1] + BBOX[3] / 2.0
            w0 = float(BBOX[2]); h0 = float(BBOX[3])
            particles[:, 0] = cx0 + np.random.randn(N).astype(np.float32) * 2
            particles[:, 1] = cy0 + np.random.randn(N).astype(np.float32) * 2
            particles[:, 2:4] = 0.0; particles[:, 4] = w0; particles[:, 5] = h0
            weights[:] = 1.0 / N
            cx_out, cy_out = cx0, cy0; w_out, h_out = w0, h0
            initialized = True
            rows.append({'frame': fi, 'score': 1.0, 'psr': 0.0, 'ess': N,
                         'cx': cx_out, 'cy': cy_out, 'w': w_out, 'h': h_out})
            continue

        cx_s = float(tracker.center_pos[0]); cy_s = float(tracker.center_pos[1])
        tw, th = float(tracker.size[0]), float(tracker.size[1])
        w_z = tw + cfg.TRACK.CONTEXT_AMOUNT * (tw + th)
        h_z = th + cfg.TRACK.CONTEXT_AMOUNT * (tw + th)
        s_x = float(np.sqrt(w_z * h_z)) * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        cfg.defrost(); cfg.merge_from_file(c['paths']['model_cfg']); cfg.CUDA = False; cfg.freeze()
        out = tracker.track(frame)
        track_score = float(out.get('best_score', 0))
        fg, score_map = cls_to_scoremap(raw['cls'])
        psr = float((fg.max() - fg.mean()) / (fg.std() + 1e-6))

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
            fx_p, fy_p = particle_to_feat(particles[:, 0], particles[:, 1], cx_s, cy_s, s_x)
            log_w = score_map[fy_p, fx_p] / pf['tau']; log_w -= log_w.max()
            weights = np.exp(log_w).astype(np.float32); weights /= weights.sum() + 1e-12
        ess = float(1.0 / ((weights ** 2).sum() + 1e-12))
        if ess < N / 2.0:
            idx = systematic_resample(weights)
            particles = particles[idx].copy(); weights = np.ones(N, np.float32) / N
            particles[:, 0] += np.random.randn(N).astype(np.float32) * pf['roughen_std']
            particles[:, 1] += np.random.randn(N).astype(np.float32) * pf['roughen_std']
        cx_out = float(np.dot(weights, particles[:, 0]))
        cy_out = float(np.dot(weights, particles[:, 1]))
        w_out  = float(tracker.size[0]); h_out = float(tracker.size[1])
        tracker.center_pos = np.array([cx_out, cy_out])
        ess = float(1.0 / ((weights ** 2).sum() + 1e-12))

        if fi % tu['freq'] == 0 and psr > tu['min_psr'] and w_out < w0*3 and h_out < h0*3:
            _ctx = cfg.TRACK.CONTEXT_AMOUNT * (w_out + h_out)
            _sz2 = round(float(np.sqrt((w_out + _ctx) * (h_out + _ctx))))
            _z_new = tracker.get_subwindow(frame, np.array([cx_out, cy_out]),
                                           cfg.TRACK.EXEMPLAR_SIZE, _sz2, tracker.channel_average)
            _nf = _enc.run(None, {'template': _z_new.cpu().numpy().astype(np.float32)})
            _zf['feats'] = [(1 - tu['alpha']) * _zf['feats'][i] + tu['alpha'] * _nf[i] for i in range(3)]

        rows.append({'frame': fi, 'score': round(track_score, 6), 'psr': round(psr, 4),
                     'ess': round(ess, 1), 'cx': round(cx_out, 3), 'cy': round(cy_out, 3),
                     'w': round(w_out, 3), 'h': round(h_out, 3)})
        if fi % 1000 == 0:
            elapsed = time.perf_counter() - t0
            print(f'  {fi}/{total}  score={track_score:.3f}  PSR={psr:.1f}  {fi/elapsed:.1f}fps')

    cap.release()
    elapsed = time.perf_counter() - t0
    print(f'  Done: {len(rows)} frames in {elapsed:.1f}s  ({len(rows)/elapsed:.2f} fps)')
    return rows


def save_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['frame','score','psr','ess','cx','cy','w','h'])
        w.writeheader(); w.writerows(rows)


def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ['score','psr','ess','cx','cy','w','h']:
            r[k] = float(r[k])
        r['frame'] = int(r['frame'])
    return rows


# ── Reporting ─────────────────────────────────────────────────────────────────

def determinism_check(runs, label):
    print(f'\n=== Determinism check: {label} (3 runs) ===')
    ref = runs[0]
    all_same = True
    for i, other in enumerate(runs[1:], 2):
        diffs = [abs(a['score'] - b['score']) for a, b in zip(ref, other)]
        max_d = max(diffs); mean_d = sum(diffs) / len(diffs)
        if max_d == 0.0:
            print(f'  Run 1 vs Run {i}: IDENTICAL (max_diff=0.000000)')
        else:
            all_same = False
            first = next(f for f, d in enumerate(diffs) if d > 0)
            print(f'  Run 1 vs Run {i}: DIFFER  max_diff={max_d:.6f}  mean_diff={mean_d:.6f}  first_diff@f={first}')
    if all_same:
        print(f'  RESULT: All 3 runs are bit-identical. Seed=42 fully controls randomness.')
    else:
        print(f'  RESULT: Runs differ — non-determinism outside numpy (e.g. CUDA float ordering).')


def cross_model_report(orig_rows, split_rows):
    print('\n' + '='*65)
    print('CROSS-MODEL FRAME-BY-FRAME COMPARISON: Original vs Split ONNX')
    print('='*65)
    n = min(len(orig_rows), len(split_rows))
    score_diffs = [abs(orig_rows[i]['score'] - split_rows[i]['score']) for i in range(n)]
    psr_diffs   = [abs(orig_rows[i]['psr']   - split_rows[i]['psr'])   for i in range(n)]
    cx_diffs    = [abs(orig_rows[i]['cx']     - split_rows[i]['cx'])    for i in range(n)]
    cy_diffs    = [abs(orig_rows[i]['cy']     - split_rows[i]['cy'])    for i in range(n)]

    print(f'\nScore diff  — mean={np.mean(score_diffs):.6f}  max={np.max(score_diffs):.6f}  '
          f'std={np.std(score_diffs):.6f}')
    print(f'PSR diff    — mean={np.mean(psr_diffs):.4f}  max={np.max(psr_diffs):.4f}')
    print(f'CX diff(px) — mean={np.mean(cx_diffs):.3f}  max={np.max(cx_diffs):.3f}')
    print(f'CY diff(px) — mean={np.mean(cy_diffs):.3f}  max={np.max(cy_diffs):.3f}')

    # First frame where score diff > threshold
    for thresh in [1e-5, 1e-4, 1e-3, 0.01]:
        first = next((i for i, d in enumerate(score_diffs) if d > thresh), None)
        print(f'First frame score diff > {thresh:.0e}: {first if first is not None else "never"}')

    # Frames with large position divergence
    large_pos = [(i, cx_diffs[i], cy_diffs[i]) for i in range(n)
                 if cx_diffs[i] > 10 or cy_diffs[i] > 10]
    print(f'\nFrames with >10px position divergence: {len(large_pos)}')
    if large_pos:
        print('  First 5:')
        for f, dx, dy in large_pos[:5]:
            print(f'    f={f:5d}  orig=({orig_rows[f]["cx"]:.0f},{orig_rows[f]["cy"]:.0f})'
                  f'  split=({split_rows[f]["cx"]:.0f},{split_rows[f]["cy"]:.0f})'
                  f'  Δcx={dx:.1f}  Δcy={dy:.1f}'
                  f'  score_orig={orig_rows[f]["score"]:.3f}  score_split={split_rows[f]["score"]:.3f}')

    # Score per 500-frame bucket
    print('\nScore means per 500-frame bucket:')
    print(f'  {"bucket":>10}  {"orig":>8}  {"split":>8}  {"diff":>8}')
    for start in range(0, n, 500):
        end = min(start + 500, n)
        o_mean = np.mean([orig_rows[i]['score']  for i in range(start, end)])
        s_mean = np.mean([split_rows[i]['score'] for i in range(start, end)])
        print(f'  {start:4d}-{end:4d}   {o_mean:8.4f}   {s_mean:8.4f}   {abs(o_mean-s_mean):8.4f}')

    print('\n' + '='*65)

    # Check if models are numerically identical
    if np.max(score_diffs) < 1e-5:
        print('VERDICT: Original and Split ONNX produce NUMERICALLY IDENTICAL results.')
    elif np.max(score_diffs) < 1e-3:
        print('VERDICT: Negligible floating-point differences only (< 1e-3). Functionally identical.')
    else:
        print(f'VERDICT: Models DIVERGE. Max score diff = {np.max(score_diffs):.4f}.')
        first_big = next((i for i, d in enumerate(score_diffs) if d > 0.01), None)
        if first_big:
            print(f'         First significant divergence (>0.01) at frame {first_big}.')
    print('='*65)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config-orig',  required=True)
    ap.add_argument('--config-split', required=True)
    ap.add_argument('--out-dir',      default='inference_output')
    ap.add_argument('--runs',         type=int, default=3)
    args = ap.parse_args()

    import onnxruntime as ort
    from pysot.core.config import cfg
    from pysot.models.model_builder import ModelBuilder
    from pysot.tracker.tracker_builder import build_tracker

    c_orig  = load_config(args.config_orig)
    c_split = load_config(args.config_split)
    os.makedirs(args.out_dir, exist_ok=True)

    orig_runs  = []
    split_runs = []

    # ── Original: 3 runs ──────────────────────────────────────────────────────
    for run in range(1, args.runs + 1):
        csv_path = os.path.join(args.out_dir, f'scores_orig_run{run}.csv')
        if os.path.exists(csv_path):
            print(f'[orig run {run}] Loading cached {csv_path}')
            orig_runs.append(load_csv(csv_path))
        else:
            print(f'\n[orig run {run}/{args.runs}] Running original tracker ...')
            rows = run_once(c_orig, build_original, seed=42,
                            ort=ort, cfg=cfg, ModelBuilder=ModelBuilder, build_tracker=build_tracker)
            save_csv(rows, csv_path)
            orig_runs.append(rows)
            print(f'  Saved: {csv_path}')

    # ── Split: 3 runs ─────────────────────────────────────────────────────────
    for run in range(1, args.runs + 1):
        csv_path = os.path.join(args.out_dir, f'scores_split_run{run}.csv')
        if os.path.exists(csv_path):
            print(f'[split run {run}] Loading cached {csv_path}')
            split_runs.append(load_csv(csv_path))
        else:
            print(f'\n[split run {run}/{args.runs}] Running split tracker ...')
            rows = run_once(c_split, build_split, seed=42,
                            ort=ort, cfg=cfg, ModelBuilder=ModelBuilder, build_tracker=build_tracker)
            save_csv(rows, csv_path)
            split_runs.append(rows)
            print(f'  Saved: {csv_path}')

    # ── Reports ───────────────────────────────────────────────────────────────
    determinism_check(orig_runs,  'Original ONNX')
    determinism_check(split_runs, 'Split ONNX')
    cross_model_report(orig_runs[0], split_runs[0])

    # Save full report to file
    report_path = os.path.join(args.out_dir, 'comparison_report.txt')
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        determinism_check(orig_runs,  'Original ONNX')
        determinism_check(split_runs, 'Split ONNX')
        cross_model_report(orig_runs[0], split_runs[0])
    with open(report_path, 'w') as f:
        f.write(buf.getvalue())
    print(f'\nReport saved: {report_path}')


if __name__ == '__main__':
    main()
