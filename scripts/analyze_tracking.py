#!/usr/bin/env python3
"""
analyze_tracking.py — re-runs the split tracker on full video, logs per-frame
scores/PSR/position to CSV, then prints a detailed report.
No video is written — runs faster.
"""
import argparse, csv, json, os, sys, time, types
import numpy as np
import cv2
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..', '..', 'pysot'))

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)


def load_config(path):
    with open(path) as f:
        return json.load(f)


def systematic_resample(weights):
    N = len(weights)
    positions = (np.random.random() + np.arange(N)) / N
    cumsum = np.cumsum(weights)
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


def build_split_tracker(c, raw_store, ort, cfg, ModelBuilder, build_tracker):
    cfg.defrost(); cfg.merge_from_file(c['paths']['model_cfg']); cfg.CUDA = False; cfg.freeze()
    model = ModelBuilder().eval()
    ckpt  = torch.load(c['paths']['checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt.get('model', ckpt)))
    tracker = build_tracker(model)

    opts = ort.SessionOptions(); opts.log_severity_level = 3
    tmpl_enc = ort.InferenceSession(c['paths']['template_encoder'], opts, providers=['CPUExecutionProvider'])
    srch_enc  = ort.InferenceSession(c['paths']['search_encoder'],   opts, providers=['CPUExecutionProvider'])
    xcorr_hd  = ort.InferenceSession(c['paths']['xcorr_head'],       opts, providers=['CPUExecutionProvider'])
    _zf = {}

    def onnx_template(self, z):
        _zf['feats'] = tmpl_enc.run(None, {'template': z.cpu().numpy().astype(np.float32)})

    def onnx_track(self, x):
        x_np = x.cpu().numpy().astype(np.float32)
        xf = srch_enc.run(None, {'search': x_np})
        zf = _zf['feats']
        cls_np, loc_np = xcorr_hd.run(None, {
            'zf_0': zf[0], 'zf_1': zf[1], 'zf_2': zf[2],
            'xf_0': xf[0], 'xf_1': xf[1], 'xf_2': xf[2],
        })
        raw_store['cls'] = cls_np
        raw_store['search_crop'] = x_np
        return {'cls': torch.from_numpy(cls_np), 'loc': torch.from_numpy(loc_np)}

    tracker.model.template = types.MethodType(onnx_template, tracker.model)
    tracker.model.track    = types.MethodType(onnx_track,    tracker.model)
    return tracker, tmpl_enc, _zf


def run_tracker(c, use_pf, csv_path):
    import onnxruntime as ort
    from pysot.core.config import cfg
    from pysot.models.model_builder import ModelBuilder
    from pysot.tracker.tracker_builder import build_tracker

    pf = c['particle_filter']
    tu = c['template_update']

    raw_store = {}
    tracker, _tmpl_enc, _zf = build_split_tracker(c, raw_store, ort, cfg, ModelBuilder, build_tracker)

    cap   = cv2.VideoCapture(c['paths']['video'])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    INIT_BBOX = c['init_bbox']

    N         = pf['n_particles']
    particles = np.zeros((N, 6), np.float32)
    weights   = np.ones(N, np.float32) / N

    cx_out = cy_out = w_out = h_out = 0.0
    cx_search = cy_search = s_x = 0.0
    fg = np.zeros(3125)
    score_map   = np.zeros((25, 25))
    track_score = 1.0
    psr = ess   = 0.0
    w0 = h0     = 0.0
    initialized = False

    rows = []
    t0 = time.perf_counter()

    for fi in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        ih, iw = frame.shape[:2]

        if not initialized:
            cfg.defrost(); cfg.merge_from_file(c['paths']['model_cfg']); cfg.CUDA = False; cfg.freeze()
            tracker.init(frame, INIT_BBOX)
            cx0, cy0 = INIT_BBOX[0] + INIT_BBOX[2] / 2.0, INIT_BBOX[1] + INIT_BBOX[3] / 2.0
            w0, h0   = float(INIT_BBOX[2]), float(INIT_BBOX[3])
            if use_pf:
                particles[:, 0] = cx0 + np.random.randn(N).astype(np.float32) * 2
                particles[:, 1] = cy0 + np.random.randn(N).astype(np.float32) * 2
                particles[:, 2:4] = 0.0
                particles[:, 4] = w0; particles[:, 5] = h0
                weights[:] = 1.0 / N
            cx_out, cy_out = cx0, cy0
            w_out, h_out   = w0, h0
            initialized = True
            rows.append({'frame': fi, 'score': 1.0, 'psr': 0.0, 'ess': 0,
                         'cx': cx_out, 'cy': cy_out, 'w': w_out, 'h': h_out})
            continue

        cx_search = float(tracker.center_pos[0])
        cy_search = float(tracker.center_pos[1])
        tw, th = float(tracker.size[0]), float(tracker.size[1])
        w_z = tw + cfg.TRACK.CONTEXT_AMOUNT * (tw + th)
        h_z = th + cfg.TRACK.CONTEXT_AMOUNT * (tw + th)
        s_x = float(np.sqrt(w_z * h_z)) * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

        cfg.defrost(); cfg.merge_from_file(c['paths']['model_cfg']); cfg.CUDA = False; cfg.freeze()
        out = tracker.track(frame)
        track_score = float(out.get('best_score', 0))

        fg, score_map = cls_to_scoremap(raw_store['cls'])
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
            w_out  = float(tracker.size[0]); h_out = float(tracker.size[1])
            tracker.center_pos = np.array([cx_out, cy_out])
            ess = float(1.0 / ((weights ** 2).sum() + 1e-12))
        else:
            bbox = out.get('bbox')
            if bbox is not None:
                cx_out = float(bbox[0] + bbox[2] / 2); cy_out = float(bbox[1] + bbox[3] / 2)
                w_out  = float(bbox[2]);                h_out  = float(bbox[3])
            else:
                cx_out = float(tracker.center_pos[0]); cy_out = float(tracker.center_pos[1])
                w_out  = float(tracker.size[0]);        h_out  = float(tracker.size[1])
            ess = float(N)

        if fi % tu['freq'] == 0 and psr > tu['min_psr'] and w_out < w0 * 3 and h_out < h0 * 3:
            _ctx  = cfg.TRACK.CONTEXT_AMOUNT * (w_out + h_out)
            _sz2  = round(float(np.sqrt((w_out + _ctx) * (h_out + _ctx))))
            _z_new = tracker.get_subwindow(frame, np.array([cx_out, cy_out]),
                                           cfg.TRACK.EXEMPLAR_SIZE, _sz2, tracker.channel_average)
            _nf = _tmpl_enc.run(None, {'template': _z_new.cpu().numpy().astype(np.float32)})
            _zf['feats'] = [(1 - tu['alpha']) * _zf['feats'][i] + tu['alpha'] * _nf[i] for i in range(3)]

        rows.append({'frame': fi, 'score': track_score, 'psr': psr, 'ess': ess,
                     'cx': cx_out, 'cy': cy_out, 'w': w_out, 'h': h_out})

        if fi % 500 == 0:
            elapsed = time.perf_counter() - t0
            fps_so_far = fi / elapsed if elapsed > 0 else 0
            print(f'  {fi}/{total}  score={track_score:.3f}  PSR={psr:.1f}  pos=({cx_out:.0f},{cy_out:.0f})  {fps_so_far:.1f}fps')

    cap.release()
    elapsed = time.perf_counter() - t0
    print(f'Done: {len(rows)} frames in {elapsed:.1f}s ({len(rows)/elapsed:.2f} fps)')

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['frame','score','psr','ess','cx','cy','w','h'])
        w.writeheader(); w.writerows(rows)
    print(f'CSV: {csv_path}')
    return rows


def report(rows_pf, rows_nopf, init_bbox):
    cx0 = init_bbox[0] + init_bbox[2] / 2.0
    cy0 = init_bbox[1] + init_bbox[3] / 2.0

    def first_below(rows, thresh):
        for r in rows[1:]:
            if r['score'] < thresh:
                return r
        return None

    def drift_dist(r):
        return np.sqrt((r['cx'] - cx0)**2 + (r['cy'] - cy0)**2)

    print('\n' + '='*65)
    print('TRACKING ANALYSIS REPORT')
    print('='*65)

    for label, rows in [('WITH PF', rows_pf), ('NO PF', rows_nopf)]:
        scores = [r['score'] for r in rows[1:]]
        psrs   = [r['psr']   for r in rows[1:]]
        total  = len(rows)

        # First drop below various thresholds
        f01  = first_below(rows, 0.1)
        f03  = first_below(rows, 0.3)
        f05  = first_below(rows, 0.5)

        # Fraction of frames above threshold
        above_09 = sum(1 for s in scores if s > 0.9) / len(scores) * 100
        above_05 = sum(1 for s in scores if s > 0.5) / len(scores) * 100
        above_03 = sum(1 for s in scores if s > 0.3) / len(scores) * 100

        # Longest consecutive run above 0.5
        best_run = cur_run = 0
        for s in scores:
            if s > 0.5: cur_run += 1; best_run = max(best_run, cur_run)
            else: cur_run = 0

        # Score percentiles
        s_arr = np.array(scores)
        p10, p25, p50, p75, p90 = np.percentile(s_arr, [10, 25, 50, 75, 90])

        print(f'\n--- {label} ({total} frames) ---')
        print(f'Score stats:  mean={np.mean(s_arr):.3f}  median={p50:.3f}  std={np.std(s_arr):.3f}')
        print(f'Percentiles:  p10={p10:.3f}  p25={p25:.3f}  p50={p50:.3f}  p75={p75:.3f}  p90={p90:.3f}')
        print(f'Above 0.9:  {above_09:.1f}%   Above 0.5:  {above_05:.1f}%   Above 0.3:  {above_03:.1f}%')
        print(f'Longest run >0.5: {best_run} frames')
        print(f'PSR: mean={np.mean(psrs):.2f}  median={np.median(psrs):.2f}  std={np.std(psrs):.2f}')
        if f05:
            print(f'First drop <0.5:  frame {f05["frame"]:5d}  score={f05["score"]:.3f}  '
                  f'pos=({f05["cx"]:.0f},{f05["cy"]:.0f})  dist_from_init={drift_dist(f05):.0f}px')
        if f03:
            print(f'First drop <0.3:  frame {f03["frame"]:5d}  score={f03["score"]:.3f}  '
                  f'pos=({f03["cx"]:.0f},{f03["cy"]:.0f})  dist_from_init={drift_dist(f03):.0f}px')
        if f01:
            print(f'First drop <0.1:  frame {f01["frame"]:5d}  score={f01["score"]:.3f}  '
                  f'pos=({f01["cx"]:.0f},{f01["cy"]:.0f})  dist_from_init={drift_dist(f01):.0f}px')
        else:
            print(f'Never dropped below 0.1')

        # Score every 500 frames
        print('Score @ checkpoints:')
        for r in rows:
            if r['frame'] % 500 == 0 and r['frame'] > 0:
                print(f'  f={r["frame"]:5d}  score={r["score"]:.3f}  PSR={r["psr"]:.1f}  '
                      f'pos=({r["cx"]:.0f},{r["cy"]:.0f})')

    print('\n--- COMPARISON ---')
    # Frame-by-frame advantage
    pf_wins  = sum(1 for a, b in zip(rows_pf[1:], rows_nopf[1:]) if a['score'] > b['score'])
    nopf_wins = sum(1 for a, b in zip(rows_pf[1:], rows_nopf[1:]) if b['score'] > a['score'])
    n = min(len(rows_pf), len(rows_nopf)) - 1
    print(f'PF higher score: {pf_wins}/{n} frames ({pf_wins/n*100:.1f}%)')
    print(f'No-PF higher score: {nopf_wins}/{n} frames ({nopf_wins/n*100:.1f}%)')

    # Divergence in position
    max_pos_diff = 0; max_diff_frame = 0
    for a, b in zip(rows_pf, rows_nopf):
        d = np.sqrt((a['cx']-b['cx'])**2 + (a['cy']-b['cy'])**2)
        if d > max_pos_diff:
            max_pos_diff = d; max_diff_frame = a['frame']
    print(f'Max position divergence between modes: {max_pos_diff:.0f}px at frame {max_diff_frame}')

    # Find first frame where positions diverge >50px
    for a, b in zip(rows_pf[1:], rows_nopf[1:]):
        d = np.sqrt((a['cx']-b['cx'])**2 + (a['cy']-b['cy'])**2)
        if d > 50:
            print(f'Positions diverge >50px at frame {a["frame"]}')
            print(f'  PF pos:    ({a["cx"]:.0f},{a["cy"]:.0f})  score={a["score"]:.3f}')
            print(f'  No-PF pos: ({b["cx"]:.0f},{b["cy"]:.0f})  score={b["score"]:.3f}')
            break
    print('='*65)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/track_config.json')
    ap.add_argument('--out-dir', default='siamrpn_metis_dec_25/final_trained/inference_output')
    args = ap.parse_args()

    c = load_config(args.config)

    pf_csv   = os.path.join(args.out_dir, 'scores_with_pf.csv')
    nopf_csv = os.path.join(args.out_dir, 'scores_no_pf.csv')

    print('=== Running WITH particle filter ===')
    rows_pf = run_tracker(c, use_pf=True,  csv_path=pf_csv)

    print('\n=== Running WITHOUT particle filter ===')
    rows_nopf = run_tracker(c, use_pf=False, csv_path=nopf_csv)

    report(rows_pf, rows_nopf, c['init_bbox'])


if __name__ == '__main__':
    main()
