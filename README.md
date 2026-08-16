# Dense Frame Annotations for Low-Resource ISL Fingerspelling Recognition

Inference code and pretrained models for the ICPR 2026 paper
**[Dense Frame Annotations for Low-Resource ISL Fingerspelling Recognition][paper]**.

[paper]: https://dl.acm.org/doi/10.1007/978-3-032-31930-2_18

- 📄 **Paper** — <https://dl.acm.org/doi/10.1007/978-3-032-31930-2_18>
- 🤗 **Dataset** — <https://huggingface.co/datasets/kirandevraj/ISL-Fingerspelling>
- 🤗 **Models** — <https://huggingface.co/kirandevraj/ISL-Fingerspelling>

Two tasks:

- **Recognition** — transcribe a pre-segmented fingerspelling clip into characters.
- **Localization** — find fingerspelling regions inside a longer video, then transcribe them.

Both published results reproduce from this repository (numbers below).

## Quick start

```bash
git clone https://github.com/Kirandevraj/ISL-Fingerspelling-Dense
cd ISL-Fingerspelling-Dense
pip install -r requirements.txt
bash demo.sh
```

`demo.sh` runs everything end to end in about a minute on a GPU, using the sample clips
included in the repo. Model weights (196 MB) download automatically from HuggingFace on
first use and are cached in `weights/`; after that everything runs offline.

CPU works too — just slower. Add `--device cpu` to force it.

Expected output:

```
$ python recognize.py --video samples/clips/-QvgvxCbm1o_seg001.mp4 --gt "vijayabaskar"
84 frames @ 30.0 fps
transcript: vijayabaskar
ground truth: vijayabaskar
CER: 0.0%

$ python localize.py --video samples/localization_window.mp4 --transcribe
599 frames @ 30.0 fps -> 1 region(s) (thr=0.95, gap=0.3s, min=0.5s)
  [   8.68 -   11.48]  conf 0.955  'vijayabaskar'
```

## Usage

```bash
python paths.py                  # check weights, samples, dependencies, GPU

# recognition -- a pre-segmented clip
python recognize.py --video clip.mp4
python recognize.py --video clip.mp4 --gt "vijayabaskar"        # also prints CER
python recognize.py --video long.mp4 --start 68.7 --end 71.5    # a slice of a video
python recognize.py --video clip.mp4 --split signer             # signer-independent model
python recognize.py --video clip.mp4 --json

# localization -- a longer video
python localize.py --video long.mp4
python localize.py --video long.mp4 --transcribe                # + transcribe each region
python localize.py --video long.mp4 --out regions.csv --scores conf.npy
python localize.py --video long.mp4 --threshold 0.9 --min-duration 0.3

# batch evaluation over your own data
python evaluate.py --manifest my_clips.csv --out preds.csv
```

`evaluate.py` takes a CSV with `video` and `transcript` columns (plus optional `uid`);
relative paths resolve against the CSV's own directory.

```csv
uid,video,transcript
clip001,clips/a.mp4,vijayabaskar
clip002,clips/b.mp4,islamabad
```

## ⚠️ Input must be signer-cropped

Frames are resized straight to 224×224 with no letterboxing, because that is how the
models were trained. Feeding wide, uncropped footage degrades accuracy badly — this is
the single most common cause of nonsense output. The bundled samples show the expected
framing: the signer fills the frame.

Any format OpenCV can decode works; frame rate is read from the file.

## Models

Downloaded on demand from [HuggingFace](https://huggingface.co/kirandevraj/ISL-Fingerspelling).

| Checkpoint | Size | What it is |
|---|---|---|
| `recognition_standard.pt` | 57.5 MB | ResNet-18 + 2-layer BiLSTM + CTC head, trained with CTC **and** frame-level cross-entropy. |
| `recognition_signer.pt` | 57.5 MB | Same, signer-independent split. |
| `frame_classifier_standard.pt` | 44.8 MB | Stage-1 ResNet-18 frame classifier, 27 classes. Drives localization. |
| `frame_classifier_signer.pt` | 44.8 MB | Same, signer-independent split. |

Recognition is 14.3M parameters. "Standard" is a random train/test split; "signer" is
signer-independent — no signer appears in both train and test, so it is the harder and
more honest number. Default is standard, matching the headline result.

Localization scores every frame with the frame classifier, takes the max softmax as
confidence, then: threshold ≥ 0.95 → merge regions separated by ≤ 0.3s → drop anything
shorter than 0.5s. Those values were grid-searched on 10 held-out videos.

To use your own checkpoints, drop them into `weights/` or set `ISLFS_WEIGHTS=/some/dir`.

## Reproducing the paper

### Recognition (Table 2)

Pulls the test split straight from HuggingFace — nothing local needed.

```bash
python reproduce_paper.py --max-frames 200                  # standard, 204 clips
python reproduce_paper.py --max-frames 200 --split signer   # signer-independent, 498 clips
python reproduce_paper.py --limit 20                        # quick check, small download
```

| ResNet-BiLSTM (RGB), Frame+Word | paper | this code |
|---|---|---|
| Standard split, 204 clips | 4.87% CER | **4.43%** |
| Signer-independent, 498 clips | 16.8% CER | **16.67%** |

Downloads ~290 MB (standard) or ~700 MB (signer) of test video, cached after the first run.

`--max-frames 200` matters: training capped sequences at 200 frames, subsampling longer
clips linearly. Without it the numbers shift.

### Localization (Table 3)

```bash
python reproduce_localization.py --video-dir /path/to/cropped_full_videos
python reproduce_localization.py --video-dir ... --limit-videos 10   # quick check
python reproduce_localization.py --video-dir ... --no-cer            # F1 only, faster
```

| RGB Frame classifier | precision | recall | F1 | CER |
|---|---|---|---|---|
| paper | 83.5% | 85.7% | 84.6% | 11.5% |
| this code (204 segments, 92 videos) | 82.8% | 85.7% | **84.2%** | **11.9%** |

Frame counts: TP 22,252 / FP 4,622 / FN 3,708 against the paper's 22,505 / 4,447 / 3,769.

**This one needs full-length videos, which are not published.** The HuggingFace repo has
the 1,308 pre-segmented clips; this evaluation runs over the 92 untrimmed source videos
those segments were cut from, since the point is finding fingerspelling inside
uncut footage. The ground truth is fetched from HuggingFace automatically; point
`--video-dir` at the signer-cropped full videos. Contact the authors for access.

Protocol, matching the original evaluation: score every frame, take a 10s window centred
on each test segment, detect inside that window, count frames. Ground truth is the main
segment **plus** any other fingerspelling annotated in the same window, so a correct extra
detection is not punished as a false positive. CER concatenates every detection
overlapping the main segment; a segment with no overlapping detection scores CER 1.0, so
missed detections are penalised rather than skipped.

### Getting the numbers to match

Three details have to follow the training code, and all three are handled by
`reproduce_paper.py`:

- **Corpus-level CER**, not per-clip — total edit distance over total reference
  characters. Per-clip averaging is printed alongside for reference.
- **Ground truth from the letter annotations**, built as training built it: lowercase,
  keep `[a-z ]`, and **do not collapse or strip whitespace**. Word boundaries are not
  always annotated as their own span, so normalising whitespace quietly changes the
  reference. Scoring against the word-level transcripts instead gives ~6.1% / ~18.0%,
  since the two disagree on 83 of 1,308 segments.
- **Spaces count** as characters in the edit distance.

`evaluate.py` deliberately uses looser normalisation (spaces stripped) because it is meant
for arbitrary user data — use `reproduce_paper.py` when comparing against the paper.

## Repository layout

```
├── demo.sh                     end-to-end smoke test
├── paths.py                    paths, weight download, self-check
├── models.py                   architectures, loaders, video decoding, CTC decode
├── recognize.py                transcribe a clip
├── localize.py                 detect regions in a video
├── evaluate.py                 batch CER over a manifest CSV
├── reproduce_paper.py          reproduce Table 2 from HuggingFace
├── reproduce_localization.py   reproduce Table 3
└── samples/                    6 test clips + a 20s window + manifest
```

## Notes

- **Localization fires on non-fingerspelling content.** That is the 78–85% precision in
  Table 3, not a bug. Segments shorter than the 0.5s minimum are dropped entirely, which
  is where recall is lost. Lower `--threshold` and `--min-duration` to trade precision
  for recall.
- **Two vocabularies.** The frame classifier has 27 outputs (space, a–z); the CTC models
  have 28 (blank at index 0). Do not mix them.
- **RGB only.** The paper's keypoint (MLP-BiLSTM) variants need an RTMPose extraction step
  and are not included here.
- **Inference only.** Training code is not part of this release.

## Citation

```bibtex
@inproceedings{islfs2026,
  title     = {Dense Frame Annotations for Low-Resource ISL Fingerspelling Recognition},
  booktitle = {International Conference on Pattern Recognition (ICPR)},
  year      = {2026},
  doi       = {10.1007/978-3-032-31930-2_18}
}
```
