"""
Everything this repo needs, resolved relative to this file.

Checkpoints are not in git. They are pulled from HuggingFace on first use and cached
in weights/ -- after that everything runs offline. Drop the files into weights/ by
hand and nothing is downloaded at all.
"""

import os
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get("ISLFS_WEIGHTS", BUNDLE / "weights"))
SAMPLES = BUNDLE / "samples"

MODEL_REPO = "kirandevraj/ISL-Fingerspelling"  # HuggingFace model repo

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
# "standard" = random train/test split; "signer" = signer-independent split
# (no signer appears in both train and test, so it is the harder, more honest number).

CHECKPOINTS = {
    # Recognition: ResNet-18 + 2-layer BiLSTM + CTC head, trained with CTC loss
    # plus frame-level cross-entropy from the dense annotations.
    # Paper Table 2: 4.87% CER standard, 16.8% CER signer-independent.
    "recognition_standard": WEIGHTS / "recognition_standard.pt",
    "recognition_signer": WEIGHTS / "recognition_signer.pt",

    # Localization: Stage-1 ResNet-18 frame classifier, 27 classes
    # (26 letters + space, no CTC blank).
    # Paper Table 3, "RGB Frame" row: 84.6% F1, 11.5% CER downstream.
    "frame_classifier_standard": WEIGHTS / "frame_classifier_standard.pt",
    "frame_classifier_signer": WEIGHTS / "frame_classifier_signer.pt",
}

# Detection hyperparameters, grid-searched on 10 held-out videos (paper §5).
DETECTION_PARAMS = {
    "rgb_frame": {"threshold": 0.95, "gap_tolerance": 0.3, "min_duration": 0.5},
    "keypoint_frame": {"threshold": 0.70, "gap_tolerance": 0.3, "min_duration": 0.5},
    "ctc_rgb": {"threshold": 0.70, "gap_tolerance": 0.7, "min_duration": 0.5},
    "ctc_keypoint": {"threshold": 0.70, "gap_tolerance": 0.7, "min_duration": 0.5},
}

# ---------------------------------------------------------------------------
# Bundled samples
# ---------------------------------------------------------------------------

SAMPLE_MANIFEST = SAMPLES / "manifest.csv"          # uid, video, transcript, split
SAMPLE_WINDOW = SAMPLES / "localization_window.mp4"  # 20s window for localize.py

# Full annotations and the other 1,300 clips are not bundled (1.8 GB of video).
# The annotations alone are public:
HF_DATASET = "https://huggingface.co/datasets/kirandevraj/ISL-Fingerspelling"


def resolve(key):
    """
    Path to a checkpoint, downloading it from HuggingFace on first use.

    Set ISLFS_WEIGHTS to point at an existing directory of checkpoints, or drop the
    files into weights/ yourself, to skip the download entirely.
    """
    path = CHECKPOINTS[key]
    if path.exists():
        return path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise FileNotFoundError(
            f"missing checkpoint: {path}\n"
            f"Install huggingface_hub to fetch it automatically (pip install -r "
            f"requirements.txt), or download {path.name} from "
            f"https://huggingface.co/{MODEL_REPO} into {WEIGHTS}/"
        ) from None

    print(f"fetching {path.name} from {MODEL_REPO} (first run only)...")
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    # local_dir puts a real file in weights/ rather than a symlink into the HF cache,
    # so the checkout stays self-contained and survives a cache wipe.
    hf_hub_download(MODEL_REPO, path.name, repo_type="model", local_dir=str(WEIGHTS))
    if not path.exists():
        raise FileNotFoundError(f"download reported success but {path} is missing")
    return path


def check():
    """Print bundle status. Exit code 0 if everything needed is present."""
    ok = True
    print(f"repo: {BUNDLE}\n")
    print(f"weights ({WEIGHTS})")
    for k, v in CHECKPOINTS.items():
        if v.exists():
            print(f"  {v.stat().st_size / 1e6:7.1f} MB  {k:28s} {v.name}")
        else:
            print(f"  on demand  {k:28s} {v.name}  <- will download on first use")

    print("\nsamples")
    for label, v in (("manifest", SAMPLE_MANIFEST), ("localization window", SAMPLE_WINDOW)):
        print(f"  {'ok  ' if v.exists() else 'MISS'}  {label:20s} {v.name}")
    clips = sorted((SAMPLES / "clips").glob("*.mp4")) if (SAMPLES / "clips").exists() else []
    print(f"  {'ok  ' if clips else 'MISS'}  {'clips':20s} {len(clips)} file(s)")

    print("\ndependencies")
    for mod in ("torch", "torchvision", "cv2", "editdistance", "tqdm"):
        try:
            m = __import__(mod)
            print(f"  ok    {mod:20s} {getattr(m, '__version__', '')}")
        except ImportError:
            ok = False
            print(f"  MISS  {mod}")

    try:
        import torch
        print(f"\ndevice: {'cuda (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'cpu only -- inference works but is slow'}")
    except ImportError:
        pass

    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if check() else 1)
