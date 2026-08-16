#!/usr/bin/env python3
"""
Fingerspelling recognition — transcribe a pre-segmented clip to characters.

This is Table 2's ResNet-BiLSTM (Frame+Word): 4.87% CER on the standard split,
16.8% CER signer-independent.

    python recognize.py --video clip.mp4
    python recognize.py --video full.mp4 --start 12.4 --end 15.1
    python recognize.py --video clip.mp4 --gt "vijayabaskar"
    python recognize.py --split signer --video clip.mp4

Input must be signer-cropped video -- see the README.
"""

import argparse
import json

import torch

import paths
from models import compute_cer, load_transcription_model, read_frames, transcribe


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="signer-cropped video file")
    ap.add_argument("--start", type=float, default=None, help="clip start (seconds)")
    ap.add_argument("--end", type=float, default=None, help="clip end (seconds)")
    ap.add_argument("--split", choices=["standard", "signer"], default="standard",
                    help="which trained model to use (default: standard)")
    ap.add_argument("--checkpoint", default=None, help="override the checkpoint path")
    ap.add_argument("--gt", default=None, help="ground-truth text; prints CER if given")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args()

    if (args.start is None) != (args.end is None):
        ap.error("--start and --end must be given together")

    ckpt = args.checkpoint or paths.resolve(f"recognition_{args.split}")

    model = load_transcription_model(ckpt, args.device)
    frames, fps = read_frames(args.video, args.start, args.end)
    text = transcribe(model, frames, args.device)

    result = {
        "video": args.video,
        "start": args.start,
        "end": args.end,
        "frames": len(frames),
        "fps": round(fps, 3),
        "checkpoint": str(ckpt),
        "transcript": text,
    }
    if args.gt:
        result["ground_truth"] = args.gt
        result["cer"] = round(compute_cer(args.gt, text), 4)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{len(frames)} frames @ {fps:.1f} fps")
        print(f"transcript: {text}")
        if args.gt:
            print(f"ground truth: {args.gt}")
            print(f"CER: {result['cer']:.1%}")


if __name__ == "__main__":
    main()
