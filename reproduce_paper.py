#!/usr/bin/env python3
"""
Recognition evaluation on the published test split, pulling the dataset from
HuggingFace. Nothing local is required except the model weights.

    python reproduce_paper.py                       # standard split, 204 clips -> Table 2
    python reproduce_paper.py --max-frames 200      # matches training exactly
    python reproduce_paper.py --split signer        # signer-independent, 498 clips
    python reproduce_paper.py --limit 20            # quick check, downloads less

Downloads <https://huggingface.co/datasets/kirandevraj/ISL-Fingerspelling> (test-split
videos only -- ~290 MB standard, ~700 MB signer) into the HuggingFace cache, then runs
the same model `recognize.py` uses.

Ground truth is reconstructed from letter_annotations.json, the dense frame
annotations, exactly as the training code built its targets.

The headline figure is corpus-level CER (total edits / total reference characters),
matching what the training loop reported. Per-clip CER is printed alongside it for
reference.
"""

import argparse
import csv
import json
from pathlib import Path

import editdistance
import torch
from tqdm import tqdm

import paths
from models import load_transcription_model, read_frames, transcribe

REPO = "kirandevraj/ISL-Fingerspelling"

# Expected test-set sizes for each split.
EXPECTED_N = {"standard": 204, "signer": 498}


def fetch_metadata(cache_dir=None):
    from huggingface_hub import hf_hub_download

    def grab(name):
        return hf_hub_download(REPO, name, repo_type="dataset", cache_dir=cache_dir)

    letters = json.load(open(grab("letter_annotations.json")))
    standard = {r["uid"]: r["split"] for r in csv.DictReader(open(grab("split_info.csv")))}
    signer = {r["uid"]: r["split"] for r in csv.DictReader(open(grab("signer_split.csv")))}
    return letters, {"standard": standard, "signer": signer}


VALID_CHARS = set("abcdefghijklmnopqrstuvwxyz ")


def letter_text(chars):
    """
    Rebuild the transcript from per-character annotations, exactly as the training
    code built its targets: lowercase, keep only [a-z ], and do NOT collapse or strip
    whitespace. Word boundaries are not always annotated as their own span, so
    normalising whitespace here would silently change the reference.
    """
    return "".join(c["letter"].lower() for c in chars if c["letter"].lower() in VALID_CHARS)


def fetch_videos(uids, cache_dir=None, workers=8):
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        REPO, repo_type="dataset", cache_dir=cache_dir, max_workers=workers,
        allow_patterns=[f"videos/{u}.mp4" for u in uids],
    )
    return Path(local) / "videos"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["standard", "signer"], default="standard")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N clips")
    ap.add_argument("--cache-dir", default=None, help="HuggingFace cache location")
    ap.add_argument("--workers", type=int, default=8, help="parallel download workers")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap sequence length, subsampling longer clips (training used 200)")
    ap.add_argument("--out", default=None, help="write per-clip predictions to CSV")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"dataset : https://huggingface.co/datasets/{REPO}")
    print(f"split   : {args.split}\n")

    print("fetching annotations...")
    letters, splits = fetch_metadata(args.cache_dir)
    assign = splits[args.split]

    # letter_annotations.json is keyed by "<uid>.mp4" in some revisions, "<uid>" in others.
    by_uid = {(k[:-4] if k.endswith(".mp4") else k): v for k, v in letters.items()}
    uids = sorted(u for u, s in assign.items() if s == "test" and u in by_uid)
    if args.limit:
        uids = uids[:args.limit]

    expected = EXPECTED_N[args.split]
    print(f"  {len(uids)} test clips" + ("" if args.limit else f" (expected: {expected})"))
    if not args.limit and len(uids) != expected:
        print(f"  WARNING: expected {expected} clips for the {args.split} split")

    print(f"\ndownloading {len(uids)} videos (cached after the first run)...")
    video_dir = fetch_videos(uids, args.cache_dir, args.workers)

    ckpt = paths.resolve(f"recognition_{args.split}")
    print(f"\nmodel   : {Path(ckpt).name}")
    print(f"device  : {args.device}\n")
    model = load_transcription_model(ckpt, args.device)

    rows, failed = [], 0
    for uid in tqdm(uids, desc="transcribing"):
        video = video_dir / f"{uid}.mp4"
        if not video.exists():
            failed += 1
            continue
        try:
            frames, _ = read_frames(video, max_frames=args.max_frames)
            hyp = transcribe(model, frames, args.device)
        except Exception as e:
            print(f"  {uid}: {e}")
            failed += 1
            continue
        rows.append({
            "uid": uid,
            "prediction": hyp,
            "ground_truth": letter_text(by_uid[uid]),
        })

    if not rows:
        print("\nno clips evaluated")
        return

    dists, lens = [], []
    for r in rows:
        # Spaces are kept, matching the training loop's compute_cer.
        ref, hyp = r["ground_truth"].lower(), r["prediction"].lower()
        if not ref:
            continue
        dists.append(editdistance.eval(ref, hyp))
        lens.append(len(ref))

    n = len(dists)
    exact = sum(1 for d in dists if d == 0)
    corpus = sum(dists) / sum(lens)
    per_clip = sum(d / l for d, l in zip(dists, lens)) / n

    print(f"\n{'=' * 68}")
    print(f" Results -- {args.split} split, {len(rows)} clips" + (f" ({failed} failed)" if failed else ""))
    print(f"{'=' * 68}")
    # The training loop reports corpus-level CER (total edits / total reference
    # characters); that is the headline figure.
    print(f"CER (corpus-level) : {corpus:.2%}   ({sum(dists)} edits / {sum(lens)} chars)")
    print(f"CER (per-clip)     : {per_clip:.2%}")
    print(f"exact matches      : {exact} / {n}  ({exact / n:.1%})")
    if args.limit:
        print(f"\n(subset of {args.limit} clips -- not the full test split)")
    elif args.max_frames is None:
        print("\nAdd --max-frames 200 to match the training-time sequence cap.")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
