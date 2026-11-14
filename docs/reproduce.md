# Reproduce the results

Commands run in a terminal inside the repo. Colab works for steps 4 and 5.

## 0. Install and run the simulated studies (no data)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,landmarks,images,plots]"
pytest -q && ruff check .
python scripts/run_synthetic_studies.py
python scripts/make_results_table.py
```

## 1. Look at your data before using it
```bash
python scripts/probe_dataset.py folder PATH_TO_UIB_FOLDER
python scripts/probe_dataset.py kdef PATH_TO_KDEF
python scripts/probe_dataset.py ckplus PATH_TO/cohn-kanade-images PATH_TO/Emotion
python scripts/probe_dataset.py fer fer2013.csv fer2013new.csv
```
The probe prints counts and a few file names only. It tells you which label and subject patterns to use.

## 1b. Study 1 on JAFFE (you supply it, 10 people, leave-one-subject-out)
```bash
python scripts/extract_features.py --source jaffe --root data/private/jaffe --crops --out data/cache/jaffe.npz
python scripts/run_study1.py --features data/cache/jaffe.npz --dataset jaffe --label "measured (local machine)" --splits 10
```

## 2. Study 1 on UIBVFEDPlus-Light (public)
```bash
python scripts/extract_features.py --source regex --root PATH --label-regex '...' \
    --group-regex '...' --label-map 'joy=happiness' --out data/cache/uib.npz --crops
python scripts/run_study1.py --features data/cache/uib.npz --dataset uibvfedplus_light \
    --label "measured (local machine)"
```

## 3. Study 1 on CK+ and KDEF (private; numbers only)
```bash
python scripts/extract_features.py --source ckplus --images-dir ... --emotion-dir ... --out data/cache/ckplus.npz
python scripts/run_study1.py --features data/cache/ckplus.npz --dataset ckplus --label "measured (local machine)"
python scripts/extract_features.py --source kdef --root ... --out data/cache/kdef.npz
python scripts/run_study1.py --features data/cache/kdef.npz --dataset kdef --label "measured (local machine)"
```
These write to `results/private/`. Check that the JSON holds no file names before committing.

## 4. Study 3 (Colab GPU for training; notebooks 04 and 05 do this from Google Drive)
Download the Kaggle zip and the FERPlus repository zip, then unpack them:
```bash
python scripts/prepare_fer_data.py --kaggle-file fer2013.tar.gz --ferplus-zip FERPlus-master.zip --out-dir data/cache/fer_data
# the printed paths are also in data/cache/fer_data/paths.json
python scripts/study3.py train --fer-csv fer2013.csv --ferplus-csv fer2013new.csv
# evaluation runs on the machine CPU, so the label says host CPU even in Colab
python scripts/study3.py evaluate --fer-csv fer2013.csv --ferplus-csv fer2013new.csv --label "measured (host CPU)"
python scripts/study3.py ood --npz data/cache/uib.npz --dataset uibvfedplus_light --label "measured (local machine)"
```

## 5. Publish the model and the demo
1. Read the pretrained weights' licence (see `docs/study3.md`).
2. Create a GitHub release and attach the ONNX file named `model.onnx`.
3. The Pages workflow copies it into the demo on the next run of the workflow.
4. Measure browser latency on a real device, add the result file with the label "measured (browser)", and only then put a number in the README.

## Before every commit
```bash
python scripts/make_dataset_table.py && python scripts/make_results_table.py
python scripts/sync_notebooks.py --check
bash scripts/preflight.sh
```
