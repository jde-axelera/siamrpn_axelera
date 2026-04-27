#!/usr/bin/env python3
"""
track_pf.py — SiamRPN++ + Particle Filter tracker
===================================================

Architecture overview
---------------------
SiamRPN++ produces a 25×25 fg-score map every frame (one ONNX forward pass).
The particle filter treats that map as its likelihood function — no extra inference.

Particle state:  s_i = (cx, cy, vx, vy, w, h)  in image-pixel coordinates.

Algorithm per frame
-------------------
1. Predict  — propagate each particle forward with a constant-velocity motion
              model + Gaussian noise (position, velocity, scale).
2. Weight   — evaluate each particle against the SiamRPN++ score map:
              wᵢ ∝ exp(score_map[fx_i, fy_i] / τ)
              Gated: skip update when PSR < 2.0 (diffuse map = no reliable signal).
3. Resample — systematic resampling when ESS = 1/Σwᵢ² drops below N/2.
4. Estimate — position from weighted mean of particles;
              size (w, h) kept from SiamRPN++ regression with shrink/growth caps.
5. Feed-back — override tracker.center_pos with PF estimate so next frame's
              search crop is centred on the PF position, not SiamRPN++'s.

Template update
---------------
Every TMPL_FREQ frames, if PSR > TMPL_PSR, blend 25% of a fresh crop of the
current target into the stored template features.  This replicates SAM2's
memory/adaptation advantage without extra models.

Output: 4-panel video
  [Original + PF bbox + particle cloud]  [Score map + particle positions]
  [Search crop]                           [SAM2 segmentation]
"""

import os, sys, types
import numpy as np
import cv2
import torch

SCRIPT_DIR = '/home/ubuntu/data/siamrpn_training'
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'pysot'))

import onnxruntime as ort
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

# ── Paths ─────────────────────────────────────────────────────────────────────
CFG_PATH  = f'{SCRIPT_DIR}/configs/config_ir_siamese.yaml'
CKPT_PATH = f'{SCRIPT_DIR}/pysot/snapshot/all_datasets_ir_siamese/best_model.pth'
ENC_PATH  = f'{SCRIPT_DIR}/exported_new_anchors/template_encoder.onnx'
TRK_PATH  = f'{SCRIPT_DIR}/exported_new_anchors/tracker.onnx'
VIDEO     = f'{SCRIPT_DIR}/ir_crop.mp4'
SAM2_VID  = f'{SCRIPT_DIR}/ir_crop_sam_seg.mp4'
OUT_PATH  = f'{SCRIPT_DIR}/ir_new_anchors_track.mp4'
INIT_BBOX = [348, 147, 38, 84]           # [x, y, w, h] — frame-0 ground truth

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)

# ── Particle filter hyper-parameters ─────────────────────────────────────────
N_PARTICLES  = 500      # number of particles
SIGMA_POS    = 4.0      # position diffusion  (px/frame)
SIGMA_VEL    = 1.5      # velocity diffusion  (px/frame²)
SIGMA_SCALE  = 0.02     # relative size noise (log-normal per frame)
MAX_VEL      = 25.0     # velocity clamp      (px/frame)
TAU          = 0.15     # likelihood temperature — lower = sharper weighting
ROUGHEN_STD  = 0.5      # px roughening added after resampling

# ── Template update hyper-parameters ─────────────────────────────────────────
TMPL_FREQ    = 30       # blend every N frames
TMPL_ALPHA   = 0.25     # blend weight: 25% new, 75% old
TMPL_PSR     = 6.0      # only update when response has a sharp focused peak


# ── Helpers ───────────────────────────────────────────────────────────────────

def systematic_resample(weights):
    """O(N) systematic resampling — lower variance than multinomial."""
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
    """(1,10,25,25) cls_raw → fg (3125,) and score_map (25,25) via max-over-anchors."""
    s = cls_raw.transpose(1, 2, 3, 0).reshape(2, -1).T        # (3125, 2)
    e = np.exp(s - s.max(axis=1, keepdims=True))
    fg = (e / e.sum(axis=1, keepdims=True))[:, 1]              # (3125,)
    score_map = fg.reshape(n_anchors, size, size).max(axis=0)  # (25,25)
    return fg, score_map


def particle_to_feat(px, py, cx_search, cy_search, s_x):
    """
    Image-space particle (px, py) → feature-map cell (fx, fy).

    PySOT anchor formula: search-patch pixel of cell (i,j) = 31.5 + i*8
    Inverse: i = (search_px - 31.5) / 8   where search_px ∈ [0, 255].
    """
    scale = 255.0 / (s_x + 1e-6)
    sx_p = (px - cx_search) * scale + 127.5
    sy_p = (py - cy_search) * scale + 127.5
    fx = np.clip(((sx_p - 31.5) / 8.0).astype(int), 0, 24)
    fy = np.clip(((sy_p - 31.5) / 8.0).astype(int), 0, 24)
    return fx, fy


def decode_search(x_np):
    img = np.clip((x_np[0].transpose(1, 2, 0) * STD + MEAN) * 255, 0, 255).astype(np.uint8)
    return img[:, :, ::-1].copy()


def add_label(img, text, bar_h=20, bg=(15, 15, 15), fg_col=(210, 210, 210)):
    bar = np.full((bar_h, img.shape[1], 3), bg, dtype=np.uint8)
    cv2.putText(bar, text, (4, bar_h - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, fg_col, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def put(img, txt, x, y, scale=0.42, col=(200, 200, 200)):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


# ── Build ONNX-backed tracker ─────────────────────────────────────────────────

def build_onnx_tracker(raw_store):
    """
    Load SiamRPN++ (PyTorch weights) then monkey-patch template() and track()
    to route through ONNX sessions.  raw_store receives cls_raw and search_crop
    each frame for the particle filter and visualisation.
    """
    cfg.defrost(); cfg.merge_from_file(CFG_PATH); cfg.CUDA = False; cfg.freeze()
    model = ModelBuilder().eval()
    ckpt  = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt.get('model', ckpt)); model.load_state_dict(sd)
    tracker = build_tracker(model)

    opts = ort.SessionOptions(); opts.log_severity_level = 3
    enc = ort.InferenceSession(ENC_PATH, opts, providers=['CPUExecutionProvider'])
    trk = ort.InferenceSession(TRK_PATH, opts, providers=['CPUExecutionProvider'])
    _zf = {}

    def onnx_template(self, z):
        _zf['feats'] = enc.run(None, {'template': z.cpu().numpy().astype(np.float32)})

    def onnx_track(self, x):
        zf   = _zf['feats']
        x_np = x.cpu().numpy().astype(np.float32)
        feed = {'zf_0': zf[0], 'zf_1': zf[1], 'zf_2': zf[2], 'search': x_np}
        cls_np, loc_np = trk.run(None, feed)
        raw_store['cls']         = cls_np
        raw_store['search_crop'] = x_np
        return {'cls': torch.from_numpy(cls_np), 'loc': torch.from_numpy(loc_np)}

    tracker.model.template = types.MethodType(onnx_template, tracker.model)
    tracker.model.track    = types.MethodType(onnx_track,    tracker.model)
    return tracker, enc, _zf


# ── Main loop ─────────────────────────────────────────────────────────────────

raw_store = {}
tracker, _enc, _zf = build_onnx_tracker(raw_store)

cap_orig = cv2.VideoCapture(VIDEO)
cap_sam  = cv2.VideoCapture(SAM2_VID)
total    = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
fps      = cap_orig.get(cv2.CAP_PROP_FPS) or 25.0
INFO_H   = 80

# Derive panel dimensions from the actual frame size (no artificial rescaling)
_, _f = cap_orig.read(); cap_orig.set(cv2.CAP_PROP_POS_FRAMES, 0)
_,  _ = cap_sam.read();  cap_sam.set(cv2.CAP_PROP_POS_FRAMES, 0)
TARGET_H = _f.shape[0]
PANEL_W  = _f.shape[1]
N_PANELS = 4
OUT_W    = PANEL_W * N_PANELS
OUT_H    = TARGET_H + 20 + INFO_H

print(f'Output: {OUT_W}x{OUT_H}  {total} frames @ {fps:.1f}fps')
writer = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (OUT_W, OUT_H))
assert writer.isOpened(), f'VideoWriter failed at {OUT_W}x{OUT_H}'


def fit_panel(img):
    """Resize img to TARGET_H height preserving aspect ratio; centre-pad to PANEL_W."""
    oh, ow = img.shape[:2]
    new_w = int(round(ow * TARGET_H / oh))
    img = cv2.resize(img, (new_w, TARGET_H))
    if new_w == PANEL_W:
        return img
    elif new_w > PANEL_W:
        x0 = (new_w - PANEL_W) // 2
        return img[:, x0:x0 + PANEL_W]
    else:
        pad = np.zeros((TARGET_H, PANEL_W, 3), dtype=np.uint8)
        x0 = (PANEL_W - new_w) // 2
        pad[:, x0:x0 + new_w] = img
        return pad


# ── State initialisation ──────────────────────────────────────────────────────
initialized  = False
particles    = np.zeros((N_PARTICLES, 6), np.float32)   # (cx,cy,vx,vy,w,h)
weights      = np.ones(N_PARTICLES,  np.float32) / N_PARTICLES
pf_cx = pf_cy = pf_w = pf_h = 0.0
cx_search = cy_search = s_x = 0.0
fg         = np.zeros(3125)
score_map  = np.zeros((25, 25))
track_score = 1.0
psr         = 0.0
ess         = float(N_PARTICLES)
search_bgr  = np.zeros((255, 255, 3), np.uint8)
sx_display  = 127.0
w0 = h0     = 0.0

for fi in range(total):
    ok_o, frame_orig = cap_orig.read()
    ok_s, frame_sam  = cap_sam.read()
    if not (ok_o and ok_s):
        break

    ih, iw = frame_orig.shape[:2]

    # ── Frame 0: initialise tracker + particles ───────────────────────────
    if not initialized:
        cfg.defrost(); cfg.merge_from_file(CFG_PATH); cfg.CUDA = False; cfg.freeze()
        tracker.init(frame_orig, INIT_BBOX)

        cx0   = INIT_BBOX[0] + INIT_BBOX[2] / 2.0
        cy0   = INIT_BBOX[1] + INIT_BBOX[3] / 2.0
        w0, h0 = float(INIT_BBOX[2]), float(INIT_BBOX[3])

        particles[:, 0] = cx0 + np.random.randn(N_PARTICLES).astype(np.float32) * 2
        particles[:, 1] = cy0 + np.random.randn(N_PARTICLES).astype(np.float32) * 2
        particles[:, 2] = 0.0   # vx
        particles[:, 3] = 0.0   # vy
        particles[:, 4] = w0
        particles[:, 5] = h0
        weights[:]       = 1.0 / N_PARTICLES

        pf_cx, pf_cy = cx0, cy0
        pf_w,  pf_h  = w0,  h0
        cx_search, cy_search = cx0, cy0
        sx_display = 127.0
        initialized = True

    else:
        # ── 1. Record search centre BEFORE tracker updates state ──────────
        cx_search = float(tracker.center_pos[0])
        cy_search = float(tracker.center_pos[1])
        tw, th    = float(tracker.size[0]), float(tracker.size[1])
        w_z = tw + cfg.TRACK.CONTEXT_AMOUNT * (tw + th)
        h_z = th + cfg.TRACK.CONTEXT_AMOUNT * (tw + th)
        s_x = float(np.sqrt(w_z * h_z)) * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        sx_display = s_x

        # ── 2. SiamRPN++ forward pass ─────────────────────────────────────
        cfg.defrost(); cfg.merge_from_file(CFG_PATH); cfg.CUDA = False; cfg.freeze()
        out = tracker.track(frame_orig)
        track_score = float(out.get('best_score', 0))

        cls_raw    = raw_store['cls']
        search_bgr = decode_search(raw_store['search_crop'])
        fg, score_map = cls_to_scoremap(cls_raw)

        # PSR: (peak − mean) / std — high = sharp isolated peak = genuine target
        psr = float((fg.max() - fg.mean()) / (fg.std() + 1e-6))

        # ── 3. Particle predict (constant-velocity + noise) ───────────────
        N = N_PARTICLES
        particles[:, 0] += particles[:, 2] + np.random.randn(N).astype(np.float32) * SIGMA_POS
        particles[:, 1] += particles[:, 3] + np.random.randn(N).astype(np.float32) * SIGMA_POS
        particles[:, 2] += np.random.randn(N).astype(np.float32) * SIGMA_VEL
        particles[:, 3] += np.random.randn(N).astype(np.float32) * SIGMA_VEL
        particles[:, 2]  = np.clip(particles[:, 2], -MAX_VEL, MAX_VEL)
        particles[:, 3]  = np.clip(particles[:, 3], -MAX_VEL, MAX_VEL)
        particles[:, 4] *= np.exp(np.random.randn(N).astype(np.float32) * SIGMA_SCALE)
        particles[:, 5] *= np.exp(np.random.randn(N).astype(np.float32) * SIGMA_SCALE)
        particles[:, 0]  = np.clip(particles[:, 0], 0, iw)
        particles[:, 1]  = np.clip(particles[:, 1], 0, ih)

        # ── 4. Likelihood weighting (PSR-gated) ───────────────────────────
        # Only update when PSR is high enough to indicate a genuine focused peak.
        # When PSR is low (diffuse map), keep previous weights and coast on
        # the motion model — prevents particles latching onto background noise.
        if psr >= 2.0:
            fx_p, fy_p = particle_to_feat(particles[:, 0], particles[:, 1],
                                          cx_search, cy_search, s_x)
            app_scores  = score_map[fy_p, fx_p]
            log_w       = app_scores / TAU
            log_w      -= log_w.max()
            weights     = np.exp(log_w).astype(np.float32)
            weights    /= weights.sum() + 1e-12

        # ── 5. Systematic resample when ESS < N/2 ─────────────────────────
        ess = float(1.0 / ((weights ** 2).sum() + 1e-12))
        if ess < N_PARTICLES / 2.0:
            idx       = systematic_resample(weights)
            particles = particles[idx].copy()
            weights   = np.ones(N_PARTICLES, np.float32) / N_PARTICLES
            particles[:, 0] += np.random.randn(N_PARTICLES).astype(np.float32) * ROUGHEN_STD
            particles[:, 1] += np.random.randn(N_PARTICLES).astype(np.float32) * ROUGHEN_STD

        # ── 6. Estimate position; keep size from SiamRPN++ ────────────────
        pf_cx = float(np.dot(weights, particles[:, 0]))
        pf_cy = float(np.dot(weights, particles[:, 1]))
        pf_w  = float(tracker.size[0])   # SiamRPN++ regression is better for size
        pf_h  = float(tracker.size[1])

        # Override tracker centre for next frame's search crop
        tracker.center_pos = np.array([pf_cx, pf_cy])

        # ── 7. PSR-gated template update (SAM2-style memory adaptation) ───
        # Blend a fresh crop of the current target into stored template features
        # every TMPL_FREQ frames when the response is a sharp, confident peak.
        _size_ok = (pf_w < w0 * 3.0 and pf_h < h0 * 3.0)
        if fi % TMPL_FREQ == 0 and psr > TMPL_PSR and _size_ok:
            _ctx  = cfg.TRACK.CONTEXT_AMOUNT * (pf_w + pf_h)
            _sz2  = round(float(np.sqrt((pf_w + _ctx) * (pf_h + _ctx))))
            _z_new = tracker.get_subwindow(frame_orig,
                                           np.array([pf_cx, pf_cy]),
                                           cfg.TRACK.EXEMPLAR_SIZE,
                                           _sz2, tracker.channel_average)
            _nf = _enc.run(None, {'template': _z_new.cpu().numpy().astype(np.float32)})
            _of = _zf['feats']
            _zf['feats'] = [(1 - TMPL_ALPHA) * _of[i] + TMPL_ALPHA * _nf[i]
                            for i in range(3)]

    # ── Build 4-panel frame ───────────────────────────────────────────────────

    # Panel 1: frame + PF bbox + search region + top-50 particles
    fvis = frame_orig.copy()
    x1 = int(pf_cx - pf_w / 2); y1 = int(pf_cy - pf_h / 2)
    cv2.rectangle(fvis, (x1, y1), (x1 + int(pf_w), y1 + int(pf_h)), (0, 255, 80), 2)
    cv2.circle(fvis,   (int(pf_cx), int(pf_cy)), 3, (0, 255, 255), -1)
    sx1 = int(pf_cx - sx_display / 2); sy1 = int(pf_cy - sx_display / 2)
    cv2.rectangle(fvis, (sx1, sy1),
                  (sx1 + int(sx_display), sy1 + int(sx_display)), (255, 100, 0), 1)
    top_idx = np.argsort(weights)[-50:]
    for pi in top_idx:
        ppx, ppy = int(particles[pi, 0]), int(particles[pi, 1])
        if 0 <= ppx < iw and 0 <= ppy < ih:
            r = max(1, int(weights[pi] * N_PARTICLES * 0.6))
            cv2.circle(fvis, (ppx, ppy), r, (0, 200, 255), -1)
    p1 = add_label(fit_panel(fvis), f'PF+SiamRPN  score={track_score:.3f}  f={fi}')

    # Panel 2: score map (JET) + particle cloud
    sq = TARGET_H
    sm_norm = ((score_map - score_map.min()) /
               (score_map.max() - score_map.min() + 1e-8) * 255).astype(np.uint8)
    hmap_big = cv2.applyColorMap(cv2.resize(sm_norm, (sq, sq),
                                            interpolation=cv2.INTER_NEAREST),
                                 cv2.COLORMAP_JET)
    if fi > 0 and s_x > 0:
        fx_vis, fy_vis = particle_to_feat(particles[:, 0], particles[:, 1],
                                          cx_search, cy_search, s_x)
        cell_px = sq / 25.0
        sx_vis  = (fx_vis * cell_px + cell_px / 2).astype(int)
        sy_vis  = (fy_vis * cell_px + cell_px / 2).astype(int)
        for pi in np.argsort(weights)[-100:]:
            if 0 <= sx_vis[pi] < sq and 0 <= sy_vis[pi] < sq:
                cv2.circle(hmap_big, (sx_vis[pi], sy_vis[pi]), 3, (255, 255, 255), -1)
    psr_col = (0, 230, 100) if psr > TMPL_PSR else (0, 80, 255)
    p2 = add_label(fit_panel(hmap_big),
                   f'Score map  PSR={psr:.1f}  ESS={ess:.0f}/{N_PARTICLES}')

    # Panel 3: search crop (255×255 normalised)
    p3 = add_label(fit_panel(search_bgr), f'Search crop  s_x={sx_display:.0f}px')

    # Panel 4: SAM2 reference
    f_sam = cv2.rotate(frame_sam, cv2.ROTATE_90_CLOCKWISE)
    p4    = add_label(fit_panel(f_sam), 'SAM2')

    # Assemble frame
    PANEL_H = TARGET_H + 20
    top_panels = []
    for p in [p1, p2, p3, p4]:
        if p.shape[:2] != (PANEL_H, PANEL_W):
            p = cv2.resize(p, (PANEL_W, PANEL_H))
        top_panels.append(p)
    top_row = np.hstack(top_panels)

    info = np.zeros((INFO_H, OUT_W, 3), np.uint8)
    score_col = (0, 255, 80)  if track_score > 0.5 else (0, 80, 255)
    ess_col   = (0, 200, 100) if ess > N_PARTICLES * 0.4 else (0, 80, 255)
    vx_mean   = float(np.dot(weights, particles[:, 2]))
    vy_mean   = float(np.dot(weights, particles[:, 3]))

    put(info, f'Frame: {fi:5d} / {total}',                    6,  18, 0.48)
    put(info, f'Track score: {track_score:.4f}',               6,  38, 0.48, score_col)
    put(info, f'PSR={psr:.1f}',                                6,  58, 0.42, psr_col)
    put(info, f'cx={pf_cx:.1f}  cy={pf_cy:.1f}',            260,  18, 0.42)
    put(info, f'w={pf_w:.1f}  h={pf_h:.1f}',                260,  38, 0.42)
    put(info, f'vel=({vx_mean:+.1f},{vy_mean:+.1f})px/f',   260,  58, 0.42)
    put(info, f'ESS={ess:.0f}/{N_PARTICLES}',                480,  18, 0.42, ess_col)
    put(info, f'N={N_PARTICLES}  σ_p={SIGMA_POS}  τ={TAU}', 480,  38, 0.42, (160, 160, 160))
    put(info, f'TMPL_PSR={TMPL_PSR}  α={TMPL_ALPHA}',       480,  58, 0.42, (160, 160, 160))

    if top_row.shape[1] != OUT_W:
        top_row = cv2.resize(top_row, (OUT_W, PANEL_H))
    frame_out = np.vstack([top_row, info])
    if frame_out.shape[:2] != (OUT_H, OUT_W):
        frame_out = cv2.resize(frame_out, (OUT_W, OUT_H))

    writer.write(frame_out)

    if fi % 500 == 0:
        print(f'  frame {fi:5d}/{total}  score={track_score:.3f}'
              f'  PSR={psr:.1f}  ESS={ess:.0f}'
              f'  pf=({pf_cx:.0f},{pf_cy:.0f})'
              f'  vel=({vx_mean:+.1f},{vy_mean:+.1f})')

cap_orig.release()
cap_sam.release()
writer.release()
print(f'\nSaved: {OUT_PATH}')
