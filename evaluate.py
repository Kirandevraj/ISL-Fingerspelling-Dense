#!/usr/bin/env python3
"""
Batch-transcribe a set of clips and report CER.

Runs on the bundled samples by default:

    python evaluate.py

To evaluate your own data, or the full 204-clip test set, point it at a CSV with
columns `video` and `transcript` (a `uid` column is used for labels if present).
Relative video paths are resolved against the CSV's own directory:

    python evaluate.py --manifest /path/to/test_set.csv

Two aggregations are printed. The paper's numbers are corpus-level (total edits /
total reference characters); per-clip averaging is shown alongside.

This script strips spaces before scoring, an appropriate default for arbitrary user
data. To compare against the publication, use reproduce_paper.py, which matches the
training code's metric exactly.
"""

import argparse
import csv
from pathlib import Path

import editdistance
import torch
from tqdm import tqdm

import paths
from models import load_transcription_model, read_frames, transcribe


def load_manifest(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    base = manifest_path.parent
    clips = []
    with open(manifest_path) as f:
        for i, row in enumerate(csv.DictReader(f)):
            if "video" not in row or "transcript" not in row:
                raise ValueError("manifest needs at least 'video' and 'transcript' columns")
            video = Path(row["video"])
            if not video.is_absolute():
                # Try the manifest's own dir, then the bundle root.
                video = next((c for c in (base / video, paths.BUNDLE / row["video"]) if c.exists()),
                             base / video)
            clips.append((row.get("uid") or f"clip{i:04d}", video, row["transcript"]))
    return clips


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(paths.SAMPLE_MANIFEST))
    ap.add_argument("--split", choices=["standard", "signer"], default="standard")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="write per-clip predictions to CSV")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = args.checkpoint or paths.resolve(f"recognition_{args.split}")
    model = load_transcription_model(ckpt, args.device)

    clips = load_manifest(args.manifest)
    if args.limit:
        clips = clips[:args.limit]
    print(f"{len(clips)} clip(s) | recognition_{args.split} | {args.device}")
    print(f"{Path(ckpt).name}\n")

    rows, tot_dist, tot_len, exact, failed = [], 0, 0, 0, 0
    for uid, video, gt in tqdm(clips, desc="transcribing", disable=len(clips) < 8):
        if not Path(video).exists():
            print(f"  {uid}: missing video {video}")
            failed += 1
            continue
        try:
            frames, _ = read_frames(video)
            hyp = transcribe(model, frames, args.device)
        except Exception as e:
            print(f"  {uid}: {e}")
            failed += 1
            continue
        r, h = gt.lower().replace(" ", ""), hyp.lower().replace(" ", "")
        d = editdistance.eval(r, h)
        tot_dist += d
        tot_len += len(r)
        exact += int(d == 0)
        rows.append({"uid": uid, "ground_truth": gt, "prediction": hyp,
                     "edit_distance": d, "ref_len": len(r),
                     "cer": round(d / max(len(r), 1), 4)})

    if not rows:
        print("\nno clips evaluated")
        return

    if len(rows) <= 20:
        print()
        for r in rows:
            mark = "  " if r["edit_distance"] == 0 else "!!"
            print(f" {mark} {r['uid']:24s} {r['prediction']!r:24s} gt={r['ground_truth']!r:20s} CER {r['cer']:6.1%}")

    mean_cer = sum(r["cer"] for r in rows) / len(rows)
    print(f"\nclips          : {len(rows)}" + (f"  ({failed} failed)" if failed else ""))
    print(f"exact matches  : {exact} ({exact / len(rows):.1%})")
    print(f"CER (per-clip) : {mean_cer:.2%}")
    print(f"CER (corpus)   : {tot_dist / max(tot_len, 1):.2%}  ({tot_dist} edits / {tot_len} chars)")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
