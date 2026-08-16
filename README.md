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

## Input requirements

Frames are resized directly to 224×224 with no letterboxing, matching the training
preprocessing. Accuracy degrades substantially on wide, uncropped footage, which is the
most common cause of poor results. The bundled samples illustrate the expected framing,
in which the signer occupies most of the frame.

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
signer-independent — no signer appears in both train and test, making it the more
demanding evaluation. The default is standard, matching the headline result.

Localization scores every frame with the frame classifier, takes the max softmax as
confidence, then: threshold ≥ 0.95 → merge regions separated by ≤ 0.3s → drop anything
shorter than 0.5s. Those values were grid-searched on 10 held-out videos.

To use your own checkpoints, place them in `weights/` or set `ISLFS_WEIGHTS=/path/to/dir`.

## Reproducing the paper

### Recognition (Table 2)

Pulls the test split straight from HuggingFace — nothing local needed.

```bash
python reproduce_paper.py --max-frames 200                  # standard, 204 clips
python reproduce_paper.py --max-frames 200 --split signer   # signer-independent, 498 clips
python reproduce_paper.py --limit 20                        # quick check, small download
```

The script reports character error rate over the published test split, both corpus-level
and averaged per clip, along with exact-match counts.

Downloads ~290 MB (standard) or ~700 MB (signer) of test video, cached after the first run.

`--max-frames 200` is required to match training, which capped sequences at 200 frames
and subsampled longer clips linearly. Omitting it shifts the reported numbers.

### Localization (Table 3)

```bash
python reproduce_localization.py --video-dir /path/to/cropped_full_videos
python reproduce_localization.py --video-dir ... --limit-videos 10   # quick check
python reproduce_localization.py --video-dir ... --no-cer            # F1 only, faster
```

The script reports frame-level precision, recall and F1 for the RGB frame classifier
across 204 segments in 92 videos, together with the downstream character error rate.

**This evaluation requires full-length videos, which are not published.** The HuggingFace repo has
the 1,308 pre-segmented clips; this evaluation runs over the 92 untrimmed source videos
those segments were cut from, as the task is to locate fingerspelling within uncut
footage. The ground truth is fetched from HuggingFace automatically; set `--video-dir`
to the directory of signer-cropped full videos.

Protocol, matching the original evaluation: score every frame, take a 10s window centred
on each test segment, detect inside that window, count frames. Ground truth is the main
segment **plus** any other fingerspelling annotated in the same window, so a correct extra
detection is not punished as a false positive. CER concatenates every detection
overlapping the main segment; a segment with no overlapping detection scores CER 1.0, so
missed detections are penalised rather than skipped.

### Getting the numbers to match

Three details must follow the training code. All three are handled by
`reproduce_paper.py`:

- **Corpus-level CER**, not per-clip — total edit distance over total reference
  characters. Per-clip averaging is printed alongside for reference.
- **Ground truth from the letter annotations**, built as training built it: lowercase,
  keep `[a-z ]`, and **do not collapse or strip whitespace**. Word boundaries are not
  always annotated as their own span, so normalising whitespace alters the reference.
  Scoring against the word-level transcripts instead gives ~6.1% / ~18.0%,
  since the two disagree on 83 of 1,308 segments.
- **Spaces count** as characters in the edit distance.

`evaluate.py` applies looser normalisation (spaces stripped), as it is intended for
arbitrary user data. Use `reproduce_paper.py` to evaluate on the published test split.

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

- **Localization produces false positives on non-fingerspelling content.** This is
  reflected in the 78–85% precision reported in Table 3. Segments shorter than the 0.5s
  minimum duration are discarded, which accounts for the loss in recall. Lower
  `--threshold` and `--min-duration` to trade precision for recall.
- **Two vocabularies.** The frame classifier has 27 outputs (space, a–z); the CTC models
  have 28 (blank at index 0). The two are not interchangeable.
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
