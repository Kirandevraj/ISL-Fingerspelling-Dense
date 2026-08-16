#!/usr/bin/env python3
"""
Fingerspelling localization — find fingerspelling regions in a full video, and
optionally transcribe each one.

This is the "RGB Frame" detector from Table 3: 84.6% F1, 11.5% CER downstream.
Per-frame confidence = max softmax of the Stage-1 frame classifier; regions come
from threshold -> gap merge -> minimum duration, with the paper's tuned values
(0.95 / 0.3s / 0.5s).

    python localize.py --video full.mp4
    python localize.py --video full.mp4 --transcribe
    python localize.py --video full.mp4 --out regions.csv --scores scores.npy

Input must be signer-cropped video -- see the README.
"""

import argparse
import csv
import json

import numpy as np
import torch

import paths
from models import (frame_confidences, load_frame_classifier,
                    load_transcription_model, read_frames, transcribe)


def detect_regions(confidences, fps, threshold, gap_tolerance, min_duration):
    """
    Binary threshold -> merge regions separated by <= gap_tolerance -> drop
    regions shorter than min_duration. Returns [(start_sec, end_sec), ...].
    """
    binary = (confidences >= threshold).astype(int)

    regions, in_region, start = [], False, 0
    for i, val in enumerate(binary):
        if val and not in_region:
            in_region, start = True, i
        elif not val and in_region:
            in_region = False
            regions.append((start / fps, (i - 1) / fps))
    if in_region:
        regions.append((start / fps, (len(confidences) - 1) / fps))

    if not regions:
        return []

    merged = [regions[0]]
    for s, e in regions[1:]:
        ps, pe = merged[-1]
        if s - pe <= gap_tolerance:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    return [(s, e) for s, e in merged if (e - s) >= min_duration]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="signer-cropped full video")
    ap.add_argument("--split", choices=["standard", "signer"], default="standard")
    ap.add_argument("--checkpoint", default=None, help="override frame classifier path")
    ap.add_argument("--transcribe", action="store_true",
                    help="also run the recognition model on each detected region")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--gap-tolerance", type=float, default=None)
    ap.add_argument("--min-duration", type=float, default=None)
    ap.add_argument("--out", default=None, help="write regions to this CSV")
    ap.add_argument("--scores", default=None, help="save per-frame confidences to this .npy")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    p = dict(paths.DETECTION_PARAMS["rgb_frame"])
    for k in ("threshold", "gap_tolerance", "min_duration"):
        if getattr(args, k) is not None:
            p[k] = getattr(args, k)

    ckpt = args.checkpoint or paths.resolve(f"frame_classifier_{args.split}")
    clf, num_classes = load_frame_classifier(ckpt, args.device)

    frames, fps = read_frames(args.video)
    conf, pred = frame_confidences(clf, frames, args.device)

    if args.scores:
        np.save(args.scores, conf)

    regions = detect_regions(conf, fps, p["threshold"], p["gap_tolerance"], p["min_duration"])

    rows = []
    trans_model = None
    if args.transcribe and regions:
        trans_model = load_transcription_model(
            paths.resolve(f"recognition_{args.split}"), args.device)

    for i, (s, e) in enumerate(regions):
        row = {
            "region": i,
            "start_sec": round(s, 3),
            "end_sec": round(e, 3),
            "duration_sec": round(e - s, 3),
            "mean_confidence": round(float(conf[int(s * fps):int(e * fps) + 1].mean()), 4),
        }
        if trans_model is not None:
            clip = frames[int(s * fps):int(e * fps) + 1]
            row["transcript"] = transcribe(trans_model, clip, args.device)
        rows.append(row)

    result = {
        "video": args.video,
        "fps": round(fps, 3),
        "total_frames": len(frames),
        "num_classes": num_classes,
        "checkpoint": str(ckpt),
        "params": p,
        "regions": rows,
    }

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["region"])
            w.writeheader()
            w.writerows(rows)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{len(frames)} frames @ {fps:.1f} fps -> {len(rows)} region(s) "
              f"(thr={p['threshold']}, gap={p['gap_tolerance']}s, min={p['min_duration']}s)")
        for r in rows:
            line = f"  [{r['start_sec']:7.2f} - {r['end_sec']:7.2f}]  conf {r['mean_confidence']:.3f}"
            if "transcript" in r:
                line += f"  {r['transcript']!r}"
            print(line)
    if args.out:
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
