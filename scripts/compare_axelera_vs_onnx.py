#!/usr/bin/env python3
"""
compare_axelera_vs_onnx.py
Run Axelera-Metis (INT8) and CPU-ONNX (FP32) trackers on the same frames
and print a frame-by-frame score/bbox comparison to find where they diverge.

Usage:
    python scripts/compare_axelera_vs_onnx.py \
        --config configs/track_config_axelera.json \
        --max-frames 200
"""
import argparse, json, os, sys, time, types
import numpy as np
import cv2
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)

# ── Axelera helpers (from track_split_axelera.py) ─────────────────────────────

class AxeleraQuantMixin:
    def _init_quant(self, model):
        self.input_infos    = model.inputs()
        self.output_infos   = model.outputs()
        self.input_info     = self.input_infos[0]
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
        self.input_buffers[0][:] = self._quantize(img_nchw)
        self.instance.run(self.input_buffers, self.output_buffers)
        return [self._dequantize(i) for i in range(len(self.output_infos))]


# ── Common helpers ─────────────────────────────────────────────────────────────

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


# ── Single-pass tracking loop ──────────────────────────────────────────────────

def run_pipeline(label, frames, c, tmpl_fn, srch_fn, xcorr_hd, tmpl_update_fn):
    """
    Run tracker for all frames, return list of per-frame stat dicts.
    tmpl_fn(z_np)  → list of 3 feature arrays (1,256,7,7) after any center-crop
    srch_fn(x_np)  → list of 3 feature arrays (1,256,31,31)
    tmpl_update_fn — same signature as tmpl_fn (may be same object)
    """
    import onnxruntime as ort
    from pysot.core.config import cfg as pysot_cfg
    from pysot.models.model_builder import ModelBuilder
    from pysot.tracker.tracker_builder import build_tracker

    np.random.seed(42)

    pysot_cfg.defrost()
    pysot_cfg.merge_from_file(c['paths']['model_cfg'])
    pysot_cfg.CUDA = False
    pysot_cfg.freeze()

    model = ModelBuilder().eval()
    ckpt  = torch.load(c['paths']['checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt.get('model', ckpt)))
    tracker = build_tracker(model)

    raw_store = {}
    _zf = {}

    def do_template(self, z):
        _zf['feats'] = tmpl_fn(z.cpu().numpy().astype(np.float32))

    def do_track(self, x):
        x_np = x.cpu().numpy().astype(np.float32)
        xf = srch_fn(x_np)
        zf = _zf['feats']
        cls_np, loc_np = xcorr_hd.run(None, {
            'zf_0': zf[0], 'zf_1': zf[1], 'zf_2': zf[2],
            'xf_0': xf[0], 'xf_1': xf[1], 'xf_2': xf[2],
        })
        raw_store['cls'] = cls_np
        return {'cls': torch.from_numpy(cls_np), 'loc': torch.from_numpy(loc_np)}

    tracker.model.template = types.MethodType(do_template, tracker.model)
    tracker.model.track    = types.MethodType(do_track,    tracker.model)

    pf     = c['particle_filter']
    tu     = c['template_update']
    use_pf = c.get('use_particle_filter', True)
    BBOX   = c['init_bbox']
    N      = pf['n_particles']

    particles = np.zeros((N, 6), np.float32)
    weights   = np.ones(N, np.float32) / N

    cx_out = cy_out = w_out = h_out = 0.0
    cx_search = cy_search = s_x = 0.0
    fg = np.zeros(3125)
    score_map   = np.zeros((25, 25))
    track_score = 1.0
    psr = 0.0
    w0 = h0 = 0.0
    initialized = False
    results = []

    for fi, frame in enumerate(frames):
        ih, iw = frame.shape[:2]

        if not initialized:
            pysot_cfg.defrost(); pysot_cfg.merge_from_file(c['paths']['model_cfg'])
            pysot_cfg.CUDA = False; pysot_cfg.freeze()
            tracker.init(frame, BBOX)
            cx0 = BBOX[0] + BBOX[2] / 2.0
            cy0 = BBOX[1] + BBOX[3] / 2.0
            w0, h0 = float(BBOX[2]), float(BBOX[3])
            if use_pf:
                particles[:, 0] = cx0 + np.random.randn(N).astype(np.float32) * 2
                particles[:, 1] = cy0 + np.random.randn(N).astype(np.float32) * 2
                particles[:, 2:4] = 0.0
                particles[:, 4] = w0; particles[:, 5] = h0
                weights[:] = 1.0 / N
            cx_out, cy_out = cx0, cy0
            w_out, h_out   = w0, h0
            initialized    = True
            results.append(dict(fi=fi, score=1.0, cx=cx_out, cy=cy_out,
                                w=w_out, h=h_out, psr=0.0, score_max=1.0))
            continue

        cx_search = float(tracker.center_pos[0])
        cy_search = float(tracker.center_pos[1])
        tw, th  = float(tracker.size[0]), float(tracker.size[1])
        w_z = tw + pysot_cfg.TRACK.CONTEXT_AMOUNT * (tw + th)
        h_z = th + pysot_cfg.TRACK.CONTEXT_AMOUNT * (tw + th)
        s_x = float(np.sqrt(w_z * h_z)) * (pysot_cfg.TRACK.INSTANCE_SIZE /
                                             pysot_cfg.TRACK.EXEMPLAR_SIZE)

        pysot_cfg.defrost(); pysot_cfg.merge_from_file(c['paths']['model_cfg'])
        pysot_cfg.CUDA = False; pysot_cfg.freeze()
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
                fx_p, fy_p = particle_to_feat(particles[:, 0], particles[:, 1],
                                              cx_search, cy_search, s_x)
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
        else:
            bbox = out.get('bbox', None)
            if bbox is not None:
                cx_out = float(bbox[0] + bbox[2] / 2)
                cy_out = float(bbox[1] + bbox[3] / 2)
                w_out, h_out = float(bbox[2]), float(bbox[3])
            else:
                cx_out = float(tracker.center_pos[0])
                cy_out = float(tracker.center_pos[1])
                w_out  = float(tracker.size[0])
                h_out  = float(tracker.size[1])

        # Template update
        if fi % tu['freq'] == 0 and psr > tu['min_psr'] and w_out < w0 * 3 and h_out < h0 * 3:
            _ctx  = pysot_cfg.TRACK.CONTEXT_AMOUNT * (w_out + h_out)
            _sz2  = round(float(np.sqrt((w_out + _ctx) * (h_out + _ctx))))
            _z_new = tracker.get_subwindow(frame, np.array([cx_out, cy_out]),
                                           pysot_cfg.TRACK.EXEMPLAR_SIZE, _sz2,
                                           tracker.channel_average)
            _nf = tmpl_update_fn(_z_new.cpu().numpy().astype(np.float32))
            _zf['feats'] = [(1 - tu['alpha']) * _zf['feats'][i] + tu['alpha'] * _nf[i]
                            for i in range(3)]

        results.append(dict(fi=fi, score=track_score, cx=cx_out, cy=cy_out,
                            w=w_out, h=h_out, psr=psr, score_max=float(fg.max())))

    print(f'  [{label}] done — {len(results)} frames')
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',          default='configs/track_config_axelera.json')
    ap.add_argument('--max-frames',      type=int, default=200)
    ap.add_argument('--template-onnx',   default='/home/ubuntu/final_trained/onnx/template_encoder.onnx')
    ap.add_argument('--search-onnx',     default='/home/ubuntu/final_trained/onnx/search_encoder.onnx')
    ap.add_argument('--csv-out',         default='inference_output/compare_axelera_vs_onnx.csv')
    args = ap.parse_args()

    import onnxruntime as ort

    c = json.load(open(args.config))
    c['max_frames'] = args.max_frames

    # ── Load frames once ──
    print(f'Loading {args.max_frames} frames from {c["paths"]["video"]} …')
    cap = cv2.VideoCapture(c['paths']['video'])
    frames = []
    for _ in range(args.max_frames):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    print(f'  Loaded {len(frames)} frames  ({frames[0].shape[1]}×{frames[0].shape[0]})')

    # ── Build shared xcorr_head session ──
    opts = ort.SessionOptions(); opts.log_severity_level = 3
    xcorr_hd = ort.InferenceSession(c['paths']['xcorr_head'], opts,
                                     providers=['CPUExecutionProvider'])

    # ── Build Axelera encoders ──
    print('\nLoading Axelera compiled models …')
    from axelera.runtime import Context
    ctx        = Context()
    tmpl_model = ctx.load_model(c['paths']['template_encoder_ax'])
    srch_model = ctx.load_model(c['paths']['search_encoder_ax'])
    tmpl_enc_ax = AxeleraEncoder(ctx, tmpl_model)
    srch_enc_ax = AxeleraEncoder(ctx, srch_model)

    def ax_tmpl(z_np):
        feats = tmpl_enc_ax.run(z_np)
        if feats[0].shape[-1] > 7:
            c7 = (feats[0].shape[-1] - 7) // 2
            feats = [f[:, :, c7:c7+7, c7:c7+7] for f in feats]
        return feats

    def ax_srch(x_np):
        return srch_enc_ax.run(x_np)

    # ── Build CPU ONNX encoders ──
    print('Loading CPU-ONNX models …')
    tmpl_enc_cpu = ort.InferenceSession(args.template_onnx, opts,
                                         providers=['CPUExecutionProvider'])
    srch_enc_cpu = ort.InferenceSession(args.search_onnx,   opts,
                                         providers=['CPUExecutionProvider'])

    def cpu_tmpl(z_np):
        return tmpl_enc_cpu.run(None, {'template': z_np})

    def cpu_srch(x_np):
        return srch_enc_cpu.run(None, {'search': x_np})

    # ── Run both pipelines ──
    print(f'\n{"="*60}')
    print(f'Run 1: Axelera (INT8/Metis) — {len(frames)} frames')
    print(f'{"="*60}')
    t0 = time.perf_counter()
    ax_results  = run_pipeline('axelera', frames, c, ax_tmpl,  ax_srch,  xcorr_hd, ax_tmpl)
    print(f'  Wall time: {time.perf_counter()-t0:.1f}s')

    print(f'\n{"="*60}')
    print(f'Run 2: CPU-ONNX (FP32) — {len(frames)} frames')
    print(f'{"="*60}')
    t0 = time.perf_counter()
    cpu_results = run_pipeline('onnx',    frames, c, cpu_tmpl, cpu_srch, xcorr_hd, cpu_tmpl)
    print(f'  Wall time: {time.perf_counter()-t0:.1f}s')

    # ── Frame-by-frame comparison ──
    print(f'\n{"="*60}')
    print(f'{"Frame":>6}  {"score_ax":>9} {"score_cpu":>10} {"Δscore":>8}  '
          f'{"cx_ax":>7} {"cx_cpu":>7} {"Δcx":>6}  '
          f'{"cy_ax":>7} {"cy_cpu":>7} {"Δcy":>6}  '
          f'{"dist":>6}  {"psr_ax":>7} {"psr_cpu":>8}')
    print('-' * 105)

    csv_rows = []
    first_large_dist = None

    for r_ax, r_cpu in zip(ax_results, cpu_results):
        fi = r_ax['fi']
        d_score = r_ax['score']  - r_cpu['score']
        d_cx    = r_ax['cx']     - r_cpu['cx']
        d_cy    = r_ax['cy']     - r_cpu['cy']
        dist    = float(np.sqrt(d_cx**2 + d_cy**2))

        csv_rows.append(dict(
            fi=fi,
            score_ax=r_ax['score'],   score_cpu=r_cpu['score'],  d_score=d_score,
            cx_ax=r_ax['cx'],         cx_cpu=r_cpu['cx'],        d_cx=d_cx,
            cy_ax=r_ax['cy'],         cy_cpu=r_cpu['cy'],        d_cy=d_cy,
            dist=dist,
            psr_ax=r_ax['psr'],       psr_cpu=r_cpu['psr'],
            score_max_ax=r_ax['score_max'], score_max_cpu=r_cpu['score_max'],
        ))

        if first_large_dist is None and dist > 5.0 and fi > 0:
            first_large_dist = fi

        flag = ' <<<' if dist > 20 else (' <' if dist > 5 else '')
        print(f'{fi:6d}  {r_ax["score"]:9.4f} {r_cpu["score"]:10.4f} {d_score:+8.4f}  '
              f'{r_ax["cx"]:7.1f} {r_cpu["cx"]:7.1f} {d_cx:+6.1f}  '
              f'{r_ax["cy"]:7.1f} {r_cpu["cy"]:7.1f} {d_cy:+6.1f}  '
              f'{dist:6.2f}  {r_ax["psr"]:7.2f} {r_cpu["psr"]:8.2f}{flag}')

    # ── Summary ──
    dists  = [r['dist']    for r in csv_rows[1:]]
    scores = [abs(r['d_score']) for r in csv_rows[1:]]
    print(f'\n{"="*60}')
    print(f'Summary ({len(frames)} frames):')
    print(f'  Position error (dist):  mean={np.mean(dists):.2f}px  '
          f'max={np.max(dists):.2f}px  p95={np.percentile(dists,95):.2f}px')
    print(f'  Score error (|Δscore|): mean={np.mean(scores):.4f}  '
          f'max={np.max(scores):.4f}  p95={np.percentile(scores,95):.4f}')
    if first_large_dist is not None:
        print(f'  First frame with dist>5px: frame {first_large_dist}')
    else:
        print(f'  No frame with dist>5px detected.')

    # ── Save CSV ──
    os.makedirs(os.path.dirname(os.path.abspath(args.csv_out)), exist_ok=True)
    import csv
    with open(args.csv_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader()
        w.writerows(csv_rows)
    print(f'\nCSV saved: {args.csv_out}')


if __name__ == '__main__':
    main()
