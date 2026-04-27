#!/usr/bin/env python3
"""
compare_axelera_sam2.py — offline video compositor
===================================================
Composes up to three pre-rendered tracking videos side by side:

    [ Axelera (no PF) | Axelera (+ PF) | MuggleSAM (ref) ]

All panels are normalised to the same height. FPS figures are read
from *_stats.json files placed alongside the input videos by the
tracker scripts, or supplied via CLI flags.

Usage — 2-panel (PF vs no-PF):
    python scripts/compare_axelera_sam2.py \
        --nopf  inference_output/ir_axelera_nopf.mp4 \
        --pf    inference_output/ir_split_axelera.mp4 \
        --output inference_output/compare_nopf_vs_pf.mp4

Usage — 3-panel (adds MuggleSAM):
    python scripts/compare_axelera_sam2.py \
        --nopf  inference_output/ir_axelera_nopf.mp4 \
        --pf    inference_output/ir_split_axelera.mp4 \
        --sam2  /path/to/ir_crop_sam_seg.mp4 \
        --output inference_output/compare_axelera_vs_sam2.mp4

The --pf video may be the full 4-panel debug output (2560px wide) — the
script automatically crops to the first tracking panel (leftmost 1/4).

FPS override (skips stats-file lookup):
    --nopf-fps 19.8 --pf-fps 19.3
"""

import argparse, json, os, subprocess, sys
import numpy as np
import cv2

LABEL_H = 28    # label bar above each panel
PANEL_H = 480   # normalised panel height


# ── Helpers ───────────────────────────────────────────────────────────────────

def fit_to_panel(img, target_h, target_w):
    """Scale-then-pad img to exactly (target_h × target_w), preserving aspect."""
    oh, ow = img.shape[:2]
    scale   = target_h / oh
    new_w   = int(round(ow * scale))
    img     = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    if new_w == target_w:
        return img
    if new_w > target_w:
        x0 = (new_w - target_w) // 2
        return img[:, x0:x0 + target_w]
    pad = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0  = (target_w - new_w) // 2
    pad[:, x0:x0 + new_w] = img
    return pad


def make_label_bar(text, w, fps=None, color=(180, 180, 180)):
    bar = np.full((LABEL_H, w, 3), (20, 20, 20), dtype=np.uint8)
    cv2.putText(bar, text, (6, LABEL_H - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
    if fps is not None:
        fps_txt = f'{fps:.2f} fps'
        (tw, _), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(bar, fps_txt, (w - tw - 8, LABEL_H - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 230, 100), 1, cv2.LINE_AA)
    return bar


def load_fps_from_stats(video_path):
    """Return fps from companion _stats.json if it exists, else None."""
    stem = os.path.splitext(video_path)[0]
    stats_path = stem + '_stats.json'
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f).get('fps')
    return None


def probe_video(path):
    """Return (width, height, frame_count, fps) via ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', path
    ]
    out = json.loads(subprocess.check_output(cmd).decode())
    s   = next(s for s in out['streams'] if s.get('codec_type') == 'video')
    w   = int(s['width'])
    h   = int(s['height'])
    n   = int(s.get('nb_frames', 0))
    num, den = s.get('r_frame_rate', '25/1').split('/')
    fps = float(num) / float(den)
    return w, h, n, fps


def crop_first_panel(frame, src_w, src_h):
    """If frame is a multi-panel composite (wider than 2× tall), crop the first panel."""
    if src_w >= src_h * 2:
        panel_w = src_w // 4 if src_w >= src_h * 4 else src_w // 2
        return frame[:, :panel_w]
    return frame


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nopf',     required=True,  help='no-PF tracking video')
    ap.add_argument('--pf',       required=True,  help='PF tracking video (may be 4-panel)')
    ap.add_argument('--sam2',     default=None,   help='SAM2 / MuggleSAM reference video')
    ap.add_argument('--output',   required=True,  help='output mp4 path')
    ap.add_argument('--nopf-fps', default=None,   type=float)
    ap.add_argument('--pf-fps',   default=None,   type=float)
    ap.add_argument('--max-frames', default=None, type=int)
    args = ap.parse_args()

    # ── Probe inputs ─────────────────────────────────────────────────────────
    nopf_w, nopf_h, nopf_n, src_fps = probe_video(args.nopf)
    pf_w,   pf_h,   pf_n,   _       = probe_video(args.pf)
    sam2_w = sam2_h = sam2_n = 0
    if args.sam2:
        sam2_w, sam2_h, sam2_n, _ = probe_video(args.sam2)

    # ── FPS values ───────────────────────────────────────────────────────────
    nopf_fps = args.nopf_fps or load_fps_from_stats(args.nopf)
    pf_fps   = args.pf_fps   or load_fps_from_stats(args.pf)

    # Determine panel width: use source video width (after any multi-panel crop)
    # no-PF video: W × (LBL+H+INFO), we crop to just the frame rows
    # PF video: may be 4-panel wide → after crop, panel_w = pf_w/4
    pf_panel_w = pf_w // 4 if pf_w >= pf_h * 4 else (pf_w // 2 if pf_w >= pf_h * 2 else pf_w)
    nopf_panel_w = nopf_w  # already single panel

    # Normalise all panels to PANEL_H height and a common width
    # Use the original frame width (before any bars) as panel width target
    # For the no-PF video, the frame area width equals nopf_w (full width)
    PANEL_W = max(pf_panel_w, nopf_panel_w)   # usually 640
    N_PANELS = 3 if args.sam2 else 2
    OUT_W    = PANEL_W * N_PANELS
    OUT_H    = LABEL_H + PANEL_H

    # ── Open video readers ────────────────────────────────────────────────────
    cap_nopf = cv2.VideoCapture(args.nopf)
    cap_pf   = cv2.VideoCapture(args.pf)
    cap_sam  = cv2.VideoCapture(args.sam2) if args.sam2 else None

    max_frames = args.max_frames or min(nopf_n, pf_n) or 9999999
    if args.sam2 and sam2_n:
        max_frames = min(max_frames, sam2_n)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUT_W}x{OUT_H}', '-pix_fmt', 'bgr24', '-r', str(src_fps),
        '-i', 'pipe:',
        '-vcodec', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
        args.output
    ]
    writer = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    nopf_label = 'Axelera  no PF'
    pf_label   = 'Axelera  + PF'
    sam2_label  = 'MuggleSAM  (reference)'

    # Label bars are the same every frame — pre-build them
    lbl_nopf = make_label_bar(nopf_label, PANEL_W, nopf_fps, color=(100, 200, 255))
    lbl_pf   = make_label_bar(pf_label,   PANEL_W, pf_fps,   color=(100, 255, 150))
    lbl_sam2 = make_label_bar(sam2_label, PANEL_W,            color=(255, 200, 100))

    # ── Frame-by-frame composition ───────────────────────────────────────────
    fi = 0
    while fi < max_frames:
        ok1, f_nopf = cap_nopf.read()
        ok2, f_pf   = cap_pf.read()
        if not ok1 or not ok2:
            break
        f_sam2 = None
        if cap_sam:
            ok3, f_sam2 = cap_sam.read()
            if not ok3:
                break

        # Extract clean frame area from each source
        # no-PF video: (LBL_H=20) + frame(H) + (INF_H=50) tall
        lbl_src = 20
        inf_src = 50
        nopf_frame_h = nopf_h - lbl_src - inf_src
        if nopf_frame_h > 0 and nopf_h > lbl_src + inf_src:
            nopf_frame = f_nopf[lbl_src:lbl_src + nopf_frame_h, :nopf_w]
        else:
            nopf_frame = f_nopf

        # PF video: crop first panel, then strip its label/info bars (same layout: 20+H+80)
        pf_raw = crop_first_panel(f_pf, pf_w, pf_h)
        pf_lbl_src = 20
        pf_inf_src = 80
        pf_frame_h = pf_h - pf_lbl_src - pf_inf_src
        if pf_frame_h > 0:
            pf_frame = pf_raw[pf_lbl_src:pf_lbl_src + pf_frame_h, :]
        else:
            pf_frame = pf_raw

        # SAM2: use as-is
        sam2_frame = f_sam2 if f_sam2 is not None else np.zeros((PANEL_H, PANEL_W, 3), np.uint8)

        # Normalise to PANEL_H × PANEL_W
        p_nopf = fit_to_panel(nopf_frame, PANEL_H, PANEL_W)
        p_pf   = fit_to_panel(pf_frame,   PANEL_H, PANEL_W)

        panels = [np.vstack([lbl_nopf, p_nopf]),
                  np.vstack([lbl_pf,   p_pf])]

        if cap_sam:
            p_sam2 = fit_to_panel(sam2_frame, PANEL_H, PANEL_W)
            panels.append(np.vstack([lbl_sam2, p_sam2]))

        row = np.hstack(panels)
        if row.shape[1] != OUT_W or row.shape[0] != OUT_H:
            row = cv2.resize(row, (OUT_W, OUT_H))

        writer.stdin.write(row.tobytes())
        fi += 1

        if fi % 200 == 0:
            print(f'  composited {fi}/{max_frames} frames')

    cap_nopf.release()
    cap_pf.release()
    if cap_sam:
        cap_sam.release()
    writer.stdin.close()
    writer.wait()

    print(f'\nSaved: {args.output}  ({fi} frames, {OUT_W}×{OUT_H})')

    # ── FPS comparison summary ────────────────────────────────────────────────
    print('\n─── FPS comparison ────────────────────────────────')
    if nopf_fps and pf_fps:
        overhead_pct = (nopf_fps - pf_fps) / nopf_fps * 100
        print(f'  Axelera no PF:  {nopf_fps:.2f} fps')
        print(f'  Axelera + PF:   {pf_fps:.2f} fps')
        print(f'  PF overhead:    {overhead_pct:+.1f}%  ({nopf_fps - pf_fps:.2f} fps delta)')
    elif nopf_fps:
        print(f'  Axelera no PF:  {nopf_fps:.2f} fps')
        print(f'  Axelera + PF:   (pass --pf-fps or add _stats.json)')
    elif pf_fps:
        print(f'  Axelera no PF:  (pass --nopf-fps or add _stats.json)')
        print(f'  Axelera + PF:   {pf_fps:.2f} fps')
    else:
        print('  (pass --nopf-fps / --pf-fps, or place _stats.json next to the videos)')
    print('───────────────────────────────────────────────────')


if __name__ == '__main__':
    main()
