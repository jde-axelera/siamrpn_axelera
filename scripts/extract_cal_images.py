#!/usr/bin/env python3
"""
extract_cal_images.py — Extract calibration frames from videos for axcompile.

Usage:
    python scripts/extract_cal_images.py \
        --videos /path/to/video1.mp4 /path/to/video2.mp4 \
        --out    cal_images/ \
        --n      400
"""
import argparse
import os
import sys
import cv2


def extract_frames(video_paths, out_dir, n_total):
    os.makedirs(out_dir, exist_ok=True)

    # Count total frames across all videos to plan stride
    cap_info = []
    for vp in video_paths:
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"Warning: cannot open {vp}, skipping", file=sys.stderr)
            cap.release()
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_info.append((vp, cap, total))

    if not cap_info:
        sys.exit("No videos could be opened.")

    grand_total = sum(t for _, _, t in cap_info)
    per_video   = max(1, n_total // len(cap_info))

    saved = 0
    for vp, cap, total in cap_info:
        quota = min(per_video, total)
        stride = max(1, total // quota)
        for i in range(quota):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * stride)
            ok, frame = cap.read()
            if not ok:
                break
            fname = os.path.join(out_dir, f"cal_{saved:05d}.jpg")
            cv2.imwrite(fname, frame)
            saved += 1
        cap.release()

    print(f"Saved {saved} calibration images to {out_dir}/")


def main():
    ap = argparse.ArgumentParser(description="Extract calibration images from video files.")
    ap.add_argument("--videos", nargs="+", required=True, help="Input video file(s)")
    ap.add_argument("--out",    default="cal_images/",    help="Output directory")
    ap.add_argument("--n",      type=int, default=400,    help="Target number of images")
    args = ap.parse_args()

    extract_frames(args.videos, args.out, args.n)


if __name__ == "__main__":
    main()
