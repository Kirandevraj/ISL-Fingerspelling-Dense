#!/usr/bin/env python3
"""
Reproduce the paper's recognition numbers from scratch, pulling the dataset from
HuggingFace. Nothing local is required except the weights in this bundle.

    python reproduce_paper.py                       # standard split, 204 clips -> Table 2
    python reproduce_paper.py --max-frames 200      # matches training exactly
    python reproduce_paper.py --split signer        # signer-independent, 498 clips
    python reproduce_paper.py --limit 20            # quick check, downloads less

Downloads <https://huggingface.co/datasets/kirandevraj/ISL-Fingerspelling> (test-split
videos only -- ~290 MB standard, ~700 MB signer) into the HuggingFace cache, then runs
the same model `recognize.py` uses.

Two ground-truth sources are reported, because they disagree:
  letter  -- reconstructed from letter_annotations.json (the dense frame annotations)
  word    -- fingerspelling_annotations.csv (the original word-level transcripts)
They differ on 83 of 1,308 segments.

The headline comparison uses corpus-level CER (total edits / total reference
characters), which is what the training loop reported and therefore what Table 2
contains. Per-clip CER is printed alongside it for reference.

Expect to land near the published numbers, not exactly on them. Training-time
evaluation ran clips in zero-padded batches; this script transcribes each clip on its
own, which is cleaner and scores slightly better. See the README.
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

# Paper Table 2, ResNet-BiLSTM (RGB), Frame+Word supervision.
PAPER_CER = {"standard": 4.87, "signer": 16.8}
PAPER_N = {"standard": 204, "signer": 498}


def fetch_metadata(cache_dir=None):
    from huggingface_hub import hf_hub_download

    def grab(name):
        return hf_hub_download(REPO, name, repo_type="dataset", cache_dir=cache_dir)

    letters = json.load(open(grab("letter_annotations.json")))
    words = {r["uid"]: r["text"] for r in csv.DictReader(open(grab("fingerspelling_annotations.csv")))}
    standard = {r["uid"]: r["split"] for r in csv.DictReader(open(grab("split_info.csv")))}
    signer = {r["uid"]: r["split"] for r in csv.DictReader(open(grab("signer_split.csv")))}
    return letters, words, {"standard": standard, "signer": signer}


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
    letters, words, splits = fetch_metadata(args.cache_dir)
    assign = splits[args.split]

    # letter_annotations.json is keyed by "<uid>.mp4" in some revisions, "<uid>" in others.
    by_uid = {(k[:-4] if k.endswith(".mp4") else k): v for k, v in letters.items()}
    uids = sorted(u for u, s in assign.items() if s == "test" and u in by_uid)
    if args.limit:
        uids = uids[:args.limit]

    expected = PAPER_N[args.split]
    print(f"  {len(uids)} test clips" + ("" if args.limit else f" (paper: {expected})"))
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
            "gt_letter": letter_text(by_uid[uid]),
            "gt_word": words.get(uid, ""),
        })

    if not rows:
        print("\nno clips evaluated")
        return

    def score(gt_key):
        dists, lens = [], []
        for r in rows:
            # Spaces are kept, matching the training loop's compute_cer.
            ref = r[gt_key].lower()
            hyp = r["prediction"].lower()
            if not ref:
                continue
            dists.append(editdistance.eval(ref, hyp))
            lens.append(len(ref))
        per_clip = sum(d / l for d, l in zip(dists, lens)) / len(dists)
        corpus = sum(dists) / sum(lens)
        exact = sum(1 for d in dists if d == 0)
        return per_clip, corpus, exact, len(dists)

    print(f"\n{'=' * 68}")
    print(f" Results -- {args.split} split, {len(rows)} clips" + (f" ({failed} failed)" if failed else ""))
    print(f"{'=' * 68}")
    print(f"{'ground truth':<16}{'CER per-clip':>14}{'CER corpus':>13}{'exact':>14}")
    for key, label in (("gt_letter", "letter annot."), ("gt_word", "word-level")):
        per_clip, corpus, exact, n = score(key)
        print(f"{label:<16}{per_clip:>13.2%}{corpus:>13.2%}{exact:>9} / {n}")

    # The training loop reports corpus-level CER (total edits / total reference
    # characters), so that is the figure to compare against the paper.
    paper = PAPER_CER[args.split]
    got = score("gt_letter")[1] * 100
    print(f"\npaper Table 2 (ResNet-BiLSTM, RGB, Frame+Word): {paper:.2f}% CER")
    print(f"this run (letter annotations, corpus-level):    {got:.2f}% CER"
          f"   [delta {got - paper:+.2f} pt]")
    if args.limit:
        print(f"\n(only {args.limit} clips -- not comparable to the published number)")
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
