#!/usr/bin/env python3
"""
compare_sam2_vs_metis.py
========================
Reads SAM2 and Metis tracking CSVs, produces:
  1. A static multi-panel plot  (compare_sam2_vs_metis.png)
  2. A side-by-side comparison video with scrolling score strip
       Left panel  — Metis + PF  (green box)
       Right panel — SAM2        (orange box)
       Bottom strip — live score/obj_score chart (last 300 frames)

Usage:
    python scripts/compare_sam2_vs_metis.py \
        --sam2     /path/to/sam2_results.csv \
        --metis    inference_output/scores_with_pf.csv \
        --video    /path/to/ir_crop.mp4 \
        --out_plot inference_output/compare_sam2_vs_metis.png \
        --out_vid  inference_output/compare_sam2_vs_metis.mp4
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── colours ──────────────────────────────────────────────────────────────────
C_METIS = (0,   220,  80)    # green  (BGR)
C_SAM2  = (30,  160, 255)    # orange (BGR)
C_METIS_PLT = '#00dc50'
C_SAM2_PLT  = '#ff9920'

STRIP_H   = 140   # height of the score strip below the two panels
PANEL_H   = 480   # height of each video panel
WINDOW    = 300   # frames shown in scrolling chart


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_metis(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                'frame': int(r['frame']),
                'score': float(r['score']),
                'psr':   float(r['psr']),
                'cx':    float(r['cx']),
                'cy':    float(r['cy']),
                'w':     float(r['w']),
                'h':     float(r['h']),
            })
    return rows


def load_sam2(path):
    """Load sam2_results.json (preferred) or sam2_results.csv as fallback."""
    # Try JSON first — it's the authoritative source with proper null handling
    json_path = os.path.splitext(path)[0] + '.json'
    if os.path.exists(json_path):
        path = json_path
        use_json = True
    else:
        use_json = False

    rows = []
    last = {'cx': 320.0, 'cy': 240.0, 'w': 40.0, 'h': 80.0}

    def parse_entry(frame, obj_score, bbox_xywh):
        score = float(obj_score) if obj_score is not None else 0.0
        if bbox_xywh is not None:
            x, y, w, h = bbox_xywh
            last.update({'cx': x + w/2, 'cy': y + h/2, 'w': w, 'h': h})
            valid = True
        else:
            valid = False
        return {
            'frame': int(frame),
            'score': score,
            'cx':    last['cx'],
            'cy':    last['cy'],
            'w':     last['w'],
            'h':     last['h'],
            'valid': valid,
        }

    with open(path) as f:
        if use_json:
            for r in json.load(f):
                rows.append(parse_entry(r['frame'], r.get('obj_score'),
                                        r.get('bbox_xywh')))
        else:
            for r in csv.DictReader(f):
                bbox = ([float(r['x']), float(r['y']), float(r['w']), float(r['h'])]
                        if r.get('x') else None)
                rows.append(parse_entry(r['frame'], r.get('obj_score') or None, bbox))

    print(f'Loaded SAM2 from {"JSON" if use_json else "CSV"}: {path}')
    n_invalid = sum(1 for r in rows if not r['valid'])
    print(f'  {len(rows)} frames, {n_invalid} frames with no detection (held at last position)')
    return rows


# ── Static plot ───────────────────────────────────────────────────────────────

def make_static_plot(metis, sam2, out_path):
    n = min(len(metis), len(sam2))
    frames = [r['frame'] for r in metis[:n]]

    m_score = [r['score'] for r in metis[:n]]
    s_score = [r['score'] for r in sam2[:n]]
    # normalise SAM2 score to [0,1] for comparison on same axis
    s_max = max(s_score) if max(s_score) > 0 else 1.0
    s_score_norm = [v / s_max for v in s_score]

    m_cx = [r['cx'] for r in metis[:n]]
    s_cx = [r['cx'] for r in sam2[:n]]
    m_cy = [r['cy'] for r in metis[:n]]
    s_cy = [r['cy'] for r in sam2[:n]]
    m_w  = [r['w']  for r in metis[:n]]
    s_w  = [r['w']  for r in sam2[:n]]
    m_h  = [r['h']  for r in metis[:n]]
    s_h  = [r['h']  for r in sam2[:n]]

    dist = [np.sqrt((mc - sc)**2 + (mcy - scy)**2)
            for mc, sc, mcy, scy in zip(m_cx, s_cx, m_cy, s_cy)]

    fig = plt.figure(figsize=(18, 14), facecolor='#0e0e0e')
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)
    ax_score = fig.add_subplot(gs[0, :])
    ax_cx    = fig.add_subplot(gs[1, 0])
    ax_cy    = fig.add_subplot(gs[1, 1])
    ax_w     = fig.add_subplot(gs[2, 0])
    ax_h     = fig.add_subplot(gs[2, 1])
    ax_dist  = fig.add_subplot(gs[3, :])

    dark_ax = dict(facecolor='#1a1a1a', grid_color='#333')

    def style(ax, title, ylabel):
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='#aaa', labelsize=8)
        for s in ax.spines.values(): s.set_color('#444')
        ax.set_title(title, color='#ddd', fontsize=10, pad=4)
        ax.set_ylabel(ylabel, color='#aaa', fontsize=8)
        ax.set_xlabel('frame', color='#aaa', fontsize=8)
        ax.grid(color='#333', linewidth=0.5)
        ax.legend(fontsize=8, facecolor='#222', edgecolor='#555',
                  labelcolor='#ccc')

    # Score
    ax_score.plot(frames, m_score,      color=C_METIS_PLT, lw=0.8, label='Metis+PF score [0–1]')
    ax_score.plot(frames, s_score_norm, color=C_SAM2_PLT,  lw=0.8, label=f'SAM2 obj_score (norm, max={s_max:.1f})', alpha=0.8)
    style(ax_score, 'Tracking confidence', 'score')

    # cx / cy
    ax_cx.plot(frames, m_cx, color=C_METIS_PLT, lw=0.8, label='Metis')
    ax_cx.plot(frames, s_cx, color=C_SAM2_PLT,  lw=0.8, label='SAM2', alpha=0.8)
    style(ax_cx, 'Centre X', 'px')

    ax_cy.plot(frames, m_cy, color=C_METIS_PLT, lw=0.8, label='Metis')
    ax_cy.plot(frames, s_cy, color=C_SAM2_PLT,  lw=0.8, label='SAM2', alpha=0.8)
    style(ax_cy, 'Centre Y', 'px')

    # w / h
    ax_w.plot(frames, m_w, color=C_METIS_PLT, lw=0.8, label='Metis')
    ax_w.plot(frames, s_w, color=C_SAM2_PLT,  lw=0.8, label='SAM2', alpha=0.8)
    style(ax_w, 'Box width', 'px')

    ax_h.plot(frames, m_h, color=C_METIS_PLT, lw=0.8, label='Metis')
    ax_h.plot(frames, s_h, color=C_SAM2_PLT,  lw=0.8, label='SAM2', alpha=0.8)
    style(ax_h, 'Box height', 'px')

    # Euclidean distance between centres
    ax_dist.plot(frames, dist, color='#ff4466', lw=0.8, label='|Metis − SAM2| centre dist (px)')
    ax_dist.axhline(np.mean(dist), color='#ff4466', lw=1, ls='--', alpha=0.6,
                    label=f'mean {np.mean(dist):.1f} px')
    style(ax_dist, 'Centre distance: Metis vs SAM2', 'px')

    fig.suptitle('SiamRPN++ Metis+PF  vs  SAM2 — ir_crop.mp4',
                 color='white', fontsize=13, y=0.99)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved static plot: {out_path}')

    # Print summary stats
    print(f'\n── Summary ({n} frames) ──────────────────────────')
    print(f'  Metis mean score : {np.mean(m_score):.4f}')
    print(f'  SAM2  mean score : {np.mean(s_score):.4f}  (raw, not normalised)')
    print(f'  Centre dist mean : {np.mean(dist):.1f} px')
    print(f'  Centre dist max  : {np.max(dist):.1f} px  (frame {frames[int(np.argmax(dist))]})')
    print(f'  Centre dist p90  : {np.percentile(dist, 90):.1f} px')


# ── Scrolling chart strip (rendered into BGR image) ───────────────────────────

def render_score_strip(frames_buf, m_scores, s_scores, width, height):
    """Draw a live score chart for the last WINDOW frames into a BGR image."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor('#111')
    ax.set_facecolor('#1a1a1a')
    ax.set_xlim(frames_buf[0], frames_buf[-1] + 1)
    ax.set_ylim(-0.05, 1.05)
    ax.plot(frames_buf, m_scores, color=C_METIS_PLT, lw=1.0, label='Metis score')
    # normalise SAM2 for same axis
    s_max = max(s_scores) if max(s_scores) > 0 else 1.0
    ax.plot(frames_buf, [v / s_max for v in s_scores],
            color=C_SAM2_PLT, lw=1.0, label=f'SAM2 score /norm')
    ax.tick_params(colors='#888', labelsize=7)
    for sp in ax.spines.values(): sp.set_color('#444')
    ax.grid(color='#2a2a2a', lw=0.5)
    ax.legend(loc='upper left', fontsize=7, facecolor='#222',
              edgecolor='#555', labelcolor='#ccc')
    fig.tight_layout(pad=0.3)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    # ensure exact size
    bgr = cv2.resize(bgr, (width, height))
    return bgr


# ── Video ─────────────────────────────────────────────────────────────────────

def draw_box(frame, cx, cy, w, h, color, label, score):
    x1 = int(cx - w / 2); y1 = int(cy - h / 2)
    x2 = int(cx + w / 2); y2 = int(cy + h / 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, (int(cx), int(cy)), 3, color, -1)
    txt = f'{label} {score:.3f}'
    cv2.putText(frame, txt, (x1, max(y1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def make_label(text, width, color_bgr):
    bar = np.full((22, width, 3), (15, 15, 15), dtype=np.uint8)
    cv2.putText(bar, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.46, color_bgr, 1, cv2.LINE_AA)
    return bar


def make_video(metis, sam2, video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Cannot open video: {video_path}'); return

    VW  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VH  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    N   = min(len(metis), len(sam2))

    PANEL_W  = VW
    OUT_W    = PANEL_W * 2
    LBL_H    = 22
    OUT_H    = LBL_H + PANEL_H + STRIP_H

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = cv2.VideoWriter(out_path,
                             cv2.VideoWriter.fourcc('m', 'p', '4', 'v'),
                             fps, (OUT_W, OUT_H))

    lbl_metis = make_label('Metis + PF  (SiamRPN++)', PANEL_W, C_METIS)
    lbl_sam2  = make_label('SAM2  (MuggleSAM reference)', PANEL_W, C_SAM2)
    labels    = np.hstack([lbl_metis, lbl_sam2])  # (22, OUT_W, 3)

    # rolling buffers for score strip
    f_buf = list(range(WINDOW))
    m_buf = [0.0] * WINDOW
    s_buf = [0.0] * WINDOW

    fi = 0
    while fi < N:
        ok, frame = cap.read()
        if not ok:
            break

        mr = metis[fi]
        sr = sam2[fi]

        # Update rolling buffers
        f_buf.append(fi)
        m_buf.append(mr['score'])
        s_buf.append(sr['score'])
        if len(f_buf) > WINDOW:
            f_buf.pop(0); m_buf.pop(0); s_buf.pop(0)

        # Left panel — Metis
        left = cv2.resize(frame.copy(), (PANEL_W, PANEL_H))
        draw_box(left, mr['cx'] * PANEL_W / VW, mr['cy'] * PANEL_H / VH,
                 mr['w']  * PANEL_W / VW, mr['h']  * PANEL_H / VH,
                 C_METIS, 'Metis', mr['score'])

        # Right panel — SAM2 (dim box when target lost)
        right = cv2.resize(frame.copy(), (PANEL_W, PANEL_H))
        box_color = C_SAM2 if sr.get('valid', True) else (80, 80, 160)
        draw_box(right,
                 sr['cx'] * PANEL_W / VW, sr['cy'] * PANEL_H / VH,
                 sr['w']  * PANEL_W / VW, sr['h']  * PANEL_H / VH,
                 box_color, 'SAM2', sr['score'])

        # Frame counter on both panels
        for img in (left, right):
            cv2.putText(img, f'f={fi}', (4, PANEL_H - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        panels = np.hstack([left, right])   # (PANEL_H, OUT_W, 3)

        # Score strip
        strip = render_score_strip(f_buf, m_buf, s_buf, OUT_W, STRIP_H)

        row = np.vstack([labels, panels, strip])  # (OUT_H, OUT_W, 3)
        writer.write(row)

        fi += 1
        if fi % 500 == 0:
            print(f'  {fi}/{N} frames')

    cap.release()
    writer.release()
    print(f'Saved video: {out_path}  ({fi} frames, {OUT_W}×{OUT_H})')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sam2',     default='sam2_results.json',
                    help='sam2_results.json (preferred) or .csv; JSON auto-detected')
    ap.add_argument('--metis',    default='inference_output/scores_with_pf.csv')
    ap.add_argument('--video',    default='../ir_crop.mp4')
    ap.add_argument('--out_plot', default='inference_output/compare_sam2_vs_metis.png')
    ap.add_argument('--out_vid',  default='inference_output/compare_sam2_vs_metis.mp4')
    ap.add_argument('--no_video', action='store_true', help='skip video generation')
    args = ap.parse_args()

    metis = load_metis(args.metis)
    sam2  = load_sam2(args.sam2)
    print(f'Loaded: Metis {len(metis)} frames, SAM2 {len(sam2)} frames')

    make_static_plot(metis, sam2, args.out_plot)

    if not args.no_video:
        make_video(metis, sam2, args.video, args.out_vid)


if __name__ == '__main__':
    main()
