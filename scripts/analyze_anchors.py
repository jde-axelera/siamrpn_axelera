#!/usr/bin/env python3
"""
analyze_anchors.py
==================
Compute data-driven anchor aspect ratios from IR drone tracking datasets.

Reads all available pysot-format annotation JSONs, collects bbox w/h values,
then fits 5 clusters in log-space to find optimal anchor ratios (h/w) for the
SiamRPN++ ANCHOR.RATIOS config field.

Usage:
    conda activate siamrpn
    python analyze_anchors.py --data_root /path/to/training/data

Results from 2026-04-23 run (288320 bboxes, anti_uav410 + dut_vtuav + dut_anti_uav
+ massmind + msrs):
    median h/w = 0.736  (IR drones mostly wider than tall)
    KMeans-5   = [0.37, 0.56, 0.79, 1.11, 2.26]
"""

import argparse
import json
import os
import numpy as np


ANNO_CANDIDATES = [
    "anti_uav410/train_pysot.json",
    "dut_vtuav/train_pysot.json",
    "dut_anti_uav/train_pysot.json",
    "massmind/train_pysot.json",
    "msrs/train_pysot.json",
    "anti_uav300/train_pysot.json",
    "vtmot/train_pysot.json",
    "birdsai/train_pysot.json",
    "hit_uav/train_pysot.json",
    "mvss/train_pysot.json",
]


def collect_ratios(data_root):
    ratios_hw = []
    for rel in ANNO_CANDIDATES:
        path = os.path.join(data_root, rel)
        if not os.path.exists(path):
            print(f"  skip (missing): {rel}")
            continue
        with open(path) as f:
            data = json.load(f)
        count = 0
        for vid_name, vid in data.items():
            for obj_id, frames in vid.items():
                if not isinstance(frames, dict):
                    continue
                for fid, bb in frames.items():
                    if isinstance(bb, list) and len(bb) == 4:
                        x1, y1, x2, y2 = bb
                        w, h = x2 - x1, y2 - y1
                        if w > 2 and h > 2:
                            ratios_hw.append(h / w)
                            count += 1
        print(f"  {os.path.dirname(rel):20s}: {count:7d} bboxes")
    return np.array(ratios_hw)


def log_kmeans(ratios, n_clusters=5, n_iter=200):
    log_r = np.log(ratios)
    # init at evenly-spaced percentiles
    pcts = np.linspace(10, 90, n_clusters)
    centers = np.log(np.percentile(ratios, pcts))
    for _ in range(n_iter):
        dists = np.abs(log_r[:, None] - centers[None, :])
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([
            log_r[labels == i].mean() if (labels == i).sum() > 0 else centers[i]
            for i in range(n_clusters)
        ])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return sorted(np.exp(centers))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/path/to/training/data")
    ap.add_argument("--n_anchors", type=int, default=5)
    args = ap.parse_args()

    print(f"Scanning {args.data_root} ...")
    ratios = collect_ratios(args.data_root)
    print(f"\nTotal bboxes : {len(ratios)}")
    print(f"mean h/w     : {ratios.mean():.3f}")
    print(f"median h/w   : {np.median(ratios):.3f}")
    print("\nPercentiles (h/w):")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  p{p:2d}: {np.percentile(ratios, p):.3f}")

    centers = log_kmeans(ratios, n_clusters=args.n_anchors)
    rounded = [round(c, 2) for c in centers]
    print(f"\nKMeans-{args.n_anchors} anchor RATIOS (h/w): {rounded}")
    print("\nPaste into config ANCHOR section:")
    print(f"    RATIOS: {rounded}")
    print(f"    ANCHOR_NUM: {args.n_anchors}")


if __name__ == "__main__":
    main()
