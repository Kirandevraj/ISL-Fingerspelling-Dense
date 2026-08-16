#!/usr/bin/env python3
"""
Localization evaluation (Table 3, "RGB Frame" row).

    python reproduce_localization.py --video-dir /path/to/cropped_full_videos
    python reproduce_localization.py --video-dir ... --limit-videos 10   # quick check

Protocol, matching the original evaluation exactly:

  1. Score every frame of the source video with the Stage-1 frame classifier;
     confidence = max softmax.
  2. For each of the annotated test segments, take a 10s window centred on it.
  3. Run threshold -> gap-merge -> min-duration detection inside that window.
  4. Ground truth = the main segment plus any *other* fingerspelling annotated in the
     same window (so a correct extra detection is not punished as a false positive).
     Other-FS regions overlapping the main segment by >= 50% of its duration are dropped
     as duplicates.
  5. Frame-level TP/FP/FN, micro-summed over all windows -> precision, recall, F1.
  6. CER: concatenate the transcripts of every detection overlapping the main segment
     and score against its word. No overlapping detection means CER = 1.0 for that
     segment, so missed detections are penalised.

Ground truth comes from HuggingFace. THE SOURCE VIDEOS DO NOT -- the HuggingFace repo
publishes the 1,308 pre-segmented clips, not the 92 full-length videos this evaluation
runs on. Point --video-dir at a local copy of the signer-cropped full videos.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import paths
from localize import detect_regions
from models import (compute_cer, frame_confidences, load_frame_classifier,
                    load_transcription_model, read_frames, transcribe)

REPO = "kirandevraj/ISL-Fingerspelling"
WINDOW_SIZE = 10.0


def fetch_ground_truth(cache_dir=None):
    from huggingface_hub import hf_hub_download

    def grab(name):
        return hf_hub_download(REPO, name, repo_type="dataset", cache_dir=cache_dir)

    segments = defaultdict(list)
    for r in csv.DictReader(open(grab("detection_eval_segments.csv"))):
        segments[r["video_id"]].append({
            "start": float(r["start_sec"]), "end": float(r["end_sec"]),
            "transcript": r["word"], "uid": r["uid"],
        })

    other_fs = defaultdict(list)
    for r in csv.DictReader(open(grab("detection_eval_other_fs.csv"))):
        other_fs[r["video_id"]].append({
            "start": float(r["start_sec"]), "end": float(r["end_sec"]),
            "transcript": r["transcript"],
        })

    return segments, other_fs


def overlap(s1, e1, s2, e2):
    return max(0.0, min(e1, e2) - max(s1, s2))


def frame_metrics(predictions, gt_regions, fps, window_start, window_end):
    n = int((window_end - window_start) * fps)
    if n <= 0:
        return 0, 0, 0

    def mask(regions):
        m = np.zeros(n, dtype=bool)
        for s, e in regions:
            a = max(0, int((s - window_start) * fps))
            b = min(n, int((e - window_start) * fps))
            m[a:b] = True
        return m

    gt = mask([(g["start"], g["end"]) for g in gt_regions])
    pred = mask(predictions)
    return int(np.sum(gt & pred)), int(np.sum(~gt & pred)), int(np.sum(gt & ~pred))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-dir", required=True,
                    help="directory of signer-cropped full-length videos, named <video_id>.mp4")
    # The detection ground truth is built from the standard-split test segments, so the
    # signer models are not a valid choice here -- they trained on some of these clips.
    ap.add_argument("--split", choices=["standard"], default="standard",
                    help=argparse.SUPPRESS)
    ap.add_argument("--limit-videos", type=int, default=None)
    ap.add_argument("--no-cer", action="store_true",
                    help="skip transcription (F1 only, much faster)")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--gap-tolerance", type=float, default=None)
    ap.add_argument("--min-duration", type=float, default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default=None, help="write per-segment results to CSV")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    p = dict(paths.DETECTION_PARAMS["rgb_frame"])
    for k in ("threshold", "gap_tolerance", "min_duration"):
        if getattr(args, k) is not None:
            p[k] = getattr(args, k)

    video_dir = Path(args.video_dir)
    if not video_dir.is_dir():
        raise SystemExit(f"--video-dir not found: {video_dir}")

    print(f"dataset : https://huggingface.co/datasets/{REPO} (ground truth)")
    print(f"videos  : {video_dir}")
    print(f"params  : threshold={p['threshold']}, gap={p['gap_tolerance']}s, "
          f"min_duration={p['min_duration']}s, window={WINDOW_SIZE}s\n")

    segments, other_fs = fetch_ground_truth(args.cache_dir)

    video_ids = sorted(v for v in segments if (video_dir / f"{v}.mp4").exists())
    missing = sorted(set(segments) - set(video_ids))
    n_seg = sum(len(segments[v]) for v in video_ids)
    print(f"{len(video_ids)} videos / {n_seg} segments available"
          + (f"  ({len(missing)} videos missing locally)" if missing else ""))
    if missing:
        print(f"  missing: {', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}")
    if args.limit_videos:
        video_ids = video_ids[:args.limit_videos]
        print(f"  limiting to {len(video_ids)} videos")

    clf, _ = load_frame_classifier(paths.resolve(f"frame_classifier_{args.split}"), args.device)
    trans = None if args.no_cer else load_transcription_model(
        paths.resolve(f"recognition_{args.split}"), args.device)

    tot_tp = tot_fp = tot_fn = 0
    cers, rows = [], []

    for video_id in tqdm(video_ids, desc="videos"):
        try:
            frames, fps = read_frames(video_dir / f"{video_id}.mp4")
        except Exception as e:
            print(f"  {video_id}: {e}")
            continue
        conf, _ = frame_confidences(clf, frames, args.device)
        video_end = len(conf) / fps

        for seg in segments[video_id]:
            centre = (seg["start"] + seg["end"]) / 2
            w_start = max(0.0, centre - WINDOW_SIZE / 2)
            w_end = min(video_end, centre + WINDOW_SIZE / 2)

            main_dur = seg["end"] - seg["start"]
            in_window = [o for o in other_fs.get(video_id, [])
                         if o["end"] > w_start and o["start"] < w_end
                         and overlap(o["start"], o["end"], seg["start"], seg["end"]) < 0.5 * main_dur]

            # Detection runs inside the window, then is shifted back to absolute time.
            a, b = max(0, int(w_start * fps)), min(len(conf), int(w_end * fps))
            preds = [(s + w_start, e + w_start) for s, e in
                     detect_regions(conf[a:b], fps, p["threshold"], p["gap_tolerance"], p["min_duration"])]

            tp, fp, fn = frame_metrics(preds, [seg] + in_window, fps, w_start, w_end)
            tot_tp, tot_fp, tot_fn = tot_tp + tp, tot_fp + fp, tot_fn + fn

            cer, hyp = None, ""
            if trans is not None:
                matched = [(s, e) for s, e in preds
                           if overlap(s, e, seg["start"], seg["end"]) > 0]
                if matched:
                    parts = []
                    for s, e in sorted(matched):
                        clip = frames[int(s * fps):int(e * fps) + 1]
                        parts.append(transcribe(trans, clip, args.device) if len(clip) else "")
                    hyp = "".join(parts)
                    cer = compute_cer(seg["transcript"], hyp)
                else:
                    cer = 1.0
                cers.append(cer)

            rows.append({
                "video_id": video_id, "uid": seg["uid"], "word": seg["transcript"],
                "start_sec": round(seg["start"], 3), "end_sec": round(seg["end"], 3),
                "other_fs_in_window": len(in_window), "num_detections": len(preds),
                "tp": tp, "fp": fp, "fn": fn,
                "prediction": hyp, "cer": None if cer is None else round(cer, 4),
            })

    if not rows:
        print("\nno segments evaluated")
        return

    precision = tot_tp / (tot_tp + tot_fp) if tot_tp + tot_fp else 0.0
    recall = tot_tp / (tot_tp + tot_fn) if tot_tp + tot_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_cer = sum(cers) / len(cers) if cers else None

    print(f"\n{'=' * 72}")
    print(f" Table 3 -- RGB Frame classifier, {len(rows)} segments / {len(video_ids)} videos")
    print(f"{'=' * 72}")
    print(f"{'':<14}{'precision':>12}{'recall':>10}{'F1':>10}{'CER':>10}")
    got_cer = "n/a" if mean_cer is None else f"{mean_cer:.1%}"
    print(f"{'result':<14}{precision:>11.1%}{recall:>10.1%}{f1:>10.1%}{got_cer:>10}")
    print(f"\nframe counts: TP {tot_tp}  FP {tot_fp}  FN {tot_fn}")

    if args.limit_videos:
        print(f"\n(subset of {args.limit_videos} videos -- not the full evaluation set)")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
