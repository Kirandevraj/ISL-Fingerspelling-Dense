#!/usr/bin/env bash
# End-to-end smoke test. Runs everything this bundle can do, in ~1 minute on a GPU.
set -euo pipefail
cd "$(dirname "$0")"

echo "=============================================================="
echo " 1. bundle check"
echo "=============================================================="
python paths.py

echo
echo "=============================================================="
echo " 2. recognition -- one clip"
echo "=============================================================="
python recognize.py --video samples/clips/-QvgvxCbm1o_seg001.mp4 --gt "vijayabaskar"

echo
echo "=============================================================="
echo " 3. recognition -- all bundled clips, with CER"
echo "=============================================================="
python evaluate.py

echo
echo "=============================================================="
echo " 4. localization -- 20s window, detect + transcribe"
echo "=============================================================="
python localize.py --video samples/localization_window.mp4 --transcribe

echo
echo "done."
