"""
Model definitions and loaders for ISL fingerspelling recognition and localization.

Lifted verbatim (architecture and preprocessing) from the experiment code so that
outputs match the paper:
  - TranscriptionModel  <- bilstm_experiment/flask_app/eval_scripts/detection_visualization/visualize_detection.py
  - FrameClassifier     <- bilstm_experiment/precompute_scores.py
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# CTC vocabulary: 0 = blank, 1 = space, 2..27 = a..z
CHAR_MAP = {0: "", 1: " ", **{i + 2: c for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}}

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Recognition: ResNet-18 + 2-layer BiLSTM + CTC head (14.3M params)
# ---------------------------------------------------------------------------

class TranscriptionModel(nn.Module):
    def __init__(self, num_classes=28, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Identity()
        self.rnn = nn.LSTM(
            input_size=512, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.ctc_fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size * 2, num_classes))
        # Present in jointly-trained checkpoints (Stage 2 keeps the frame CE branch).
        self.frame_classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))

    def forward(self, frames):
        """frames: (T, 3, 224, 224) -> per-frame CTC logits (T, num_classes)"""
        feats = self.cnn(frames).unsqueeze(0)
        rnn_out, _ = self.rnn(feats)
        return self.ctc_fc(rnn_out).squeeze(0)


def load_transcription_model(checkpoint_path, device="cuda"):
    model = TranscriptionModel(num_classes=28).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = dict(ckpt.get("model_state_dict", ckpt))

    # The word-only runs (exp_1_1) name the CTC head `fc`; joint runs name it `ctc_fc`.
    if not any(k.startswith("ctc_fc.") for k in state_dict):
        state_dict = {("ctc_fc." + k[3:] if k.startswith("fc.") else k): v
                      for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # Word-only checkpoints (exp_1_1) have no frame_classifier branch; that is fine
        # for transcription, but flag anything else.
        unexpected_misses = [k for k in missing if not k.startswith("frame_classifier")]
        if unexpected_misses:
            raise RuntimeError(f"checkpoint is missing required weights: {unexpected_misses[:5]}")
    model.eval()
    return model


def ctc_greedy_decode(logits):
    """Collapse repeats, drop blanks. logits: (T, num_classes)"""
    pred_ids = F.softmax(logits, dim=-1).argmax(dim=-1).cpu().numpy()
    out, prev = [], -1
    for idx in pred_ids:
        if idx != prev and idx != 0:
            out.append(CHAR_MAP.get(int(idx), ""))
        prev = idx
    return "".join(out).strip()


# ---------------------------------------------------------------------------
# Localization: Stage-1 ResNet-18 frame classifier
# ---------------------------------------------------------------------------

def load_frame_classifier(checkpoint_path, device="cuda"):
    """
    Returns (model, num_classes). The released Stage-1 classifier has 27 classes
    (26 letters + space, no blank) and a bare nn.Linear head; joint-training runs
    save a Dropout+Linear head instead, so both layouts are handled.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    has_dropout = any(k.startswith(("fc.0", "fc.1")) for k in state_dict)
    num_classes = state_dict["fc.1.weight" if has_dropout else "fc.weight"].shape[0]

    model = models.resnet18(weights=None)
    model.fc = (nn.Sequential(nn.Dropout(0.3), nn.Linear(512, num_classes))
                if has_dropout else nn.Linear(512, num_classes))
    model.load_state_dict(state_dict)
    return model.to(device).eval(), num_classes


# ---------------------------------------------------------------------------
# Video I/O — must match training preprocessing exactly
# ---------------------------------------------------------------------------

def read_frames(video_path, start_sec=None, end_sec=None, max_frames=None):
    """
    Decode a video (or a time slice of one) to a (T, 3, 224, 224) float tensor.

    NOTE: frames are resized straight to 224x224 with no letterboxing, because the
    models were trained on signer-cropped video. Feed uncropped footage and accuracy
    collapses -- crop first -- see the README.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_wanted = None
    if start_sec is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
        n_wanted = int(end_sec * fps) - int(start_sec * fps) + 1

    frames = []
    while n_wanted is None or len(frames) < n_wanted:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(IMAGENET_TRANSFORM(frame))
    cap.release()

    if not frames:
        raise ValueError(f"no frames decoded from {video_path}")

    # Training capped sequences at 200 frames, linearly subsampling anything longer.
    # Pass max_frames=200 to match that; leave it None to use every frame.
    if max_frames is not None and len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]

    return torch.stack(frames), fps


@torch.no_grad()
def transcribe(model, frames, device="cuda", batch_size=32):
    """frames: (T, 3, 224, 224) -> decoded string"""
    frames = frames.to(device)
    feats = torch.cat([model.cnn(frames[i:i + batch_size])
                       for i in range(0, len(frames), batch_size)], dim=0)
    rnn_out, _ = model.rnn(feats.unsqueeze(0))
    return ctc_greedy_decode(model.ctc_fc(rnn_out).squeeze(0))


@torch.no_grad()
def frame_confidences(model, frames, device="cuda", batch_size=64):
    """
    frames: (T, 3, 224, 224) -> (confidences, predictions), both length T.

    Confidence is the max softmax probability, exactly as used for the "RGB Frame"
    detector in Table 3.
    """
    conf, pred = [], []
    for i in range(0, len(frames), batch_size):
        probs = torch.softmax(model(frames[i:i + batch_size].to(device)), dim=1)
        p, idx = probs.max(dim=1)
        conf.append(p.cpu().numpy())
        pred.append(idx.cpu().numpy())
    return np.concatenate(conf), np.concatenate(pred)


def compute_cer(reference, hypothesis):
    import editdistance
    ref = reference.lower().replace(" ", "")
    hyp = hypothesis.lower().replace(" ", "")
    if not ref:
        return 1.0 if hyp else 0.0
    return editdistance.eval(ref, hyp) / len(ref)
