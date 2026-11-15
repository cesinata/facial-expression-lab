# Decisions

Defaults chosen where the project brief left a choice. Change one by editing this file and telling me.

| Date | Decision | Reason |
|---|---|---|
| 2026-11-08 | New repo from scratch, no history from the earlier repos. | The earlier repos contain files derived from restricted datasets. |
| 2026-11-08 | Code under MIT. Datasets and third-party models keep their own terms. | Standard for a portfolio repo. |
| 2026-11-08 | Study 1 primary data: UIBVFEDPlus-Light, grouped by character. | CC BY 4.0, many subjects, no faces of real people. Download is 33.7 GB, so a partial download or subset is checked in Phase 2. |
| 2026-11-08 | JAFFE is supplied by the user, used for evaluation only, never downloaded by code. | The provider forbids redistribution, including via Colab. |
| 2026-11-08 | FER2013 is downloaded by the user's own Kaggle account. Terms still to be read. | Terms unconfirmed. Superseded on 2026-11-15 (terms read, see below). |
| 2026-11-08 | Grouped 5-fold with an inner group-aware validation split. | Keeps the test fold untouched. |
| 2026-11-08 | No code derived from the research-placement repo. Study 2 uses synthetic data. | The repo was not available and the real data is lab-owned student data. |
| 2026-11-08 | Private-dataset numbers come from scripts run on the owner's machine, committed as JSON with numbers only. | The datasets cannot be published. |
| 2026-11-08 | Every metric comes from a run by the owner. Until then the README says "not measured". | No claim without output. |
| 2026-11-08 | `scripts/preflight.sh` reads extra banned words from `.git/info/preflight-patterns`, not from a committed list. | A committed list of words to avoid would itself name them. |
| 2026-11-08 | CI actions pinned to major version tags (checkout v4, setup-python v5). Latest versions were not checked. | No network access when written. Check before release. |
| 2026-11-11 | New AI-log entries are drafted in `private/ai-log-drafts.md` (ignored by git) and moved into `docs/ai-log.md` after review. | Review happens at the end of the project. The Phase 1 entries already in git history stay as they are. |
| 2026-11-11 | Study 1 uses 40 documented landmark indices and all 780 pairwise distances, normalised by inter-ocular distance. | The thesis pair lists were not available. Region claims need a visual check first (notebook 03). |
| 2026-11-11 | The 1D CNN is a re-implementation of the thesis block structure, with the best-validation checkpoint reloaded. | Fixes a known flaw in the thesis code. It is not the thesis model. |
| 2026-11-11 | UIBVFEDPlus-Light label and subject patterns are supplied as regexes after running the probe script. | The folder layout was not known when the loader was written. |
| 2026-11-11 | CK+ runs use only provider-labelled sequences. Hand-labelled sequences, if used, are a separate run. | About a quarter of the sequences in the earlier work had no provider label. |
| 2026-11-11 | The browser demo uses the FP32 ONNX model unless INT8 has been run in the browser. | ONNX Runtime Web may not support every INT8 operator. |
| 2026-11-11 | Demo CDN versions: onnxruntime-web 1.30.0 and @mediapipe/tasks-vision 0.10.14. The first was seen on a release listing about a month old. The second comes from memory, not a lookup. | Pinned to exact versions at least two weeks old. Verify both before release. |
| 2026-11-11 | GitHub Actions versions (checkout v4, setup-python v5, configure-pages v5, upload-pages-artifact v3, deploy-pages v4) are major tags from memory. | No network when written. Check before release. |
| 2026-11-14 | FER+ licence read from the repository LICENSE.md: MIT. FER2013 images keep the Kaggle terms, which are still unread. | Recorded in `datasets.yaml`. The Kaggle part is superseded on 2026-11-15 (see below). |
| 2026-11-11 | Latency is measured one sample at a time and labelled with where it ran. | Batch throughput and single-sample latency differ. |
| 2026-11-15 | The published model is the 40-epoch run, chosen by macro F1 on PublicTest (0.590 against 0.515 for 15 epochs). | PrivateTest was read once, after this choice. |
| 2026-11-15 | `model_int8.onnx` is not published. The INT8 row stays in the results as a measured failure (accuracy 0.047). | Dynamic quantisation damaged the model. |
| 2026-11-15 | The model card is checked against `results/*.json` by `tests/test_model_card.py`. The results table adds notes only when the files show the condition. | Keeps hand-written numbers from drifting from the measured ones. |
| 2026-11-15 | Kaggle FER2013 terms read by the repo owner. The Data tab lists the licence as "Subject to Competition Rules". The rules forbid private sharing of code or data outside teams and require additional training data to be publicly available. They do not say whether models trained on the data may be published. | Recorded in `datasets.yaml` as `partly_confirmed`: the terms were read, but they do not cover the question that matters for the release. |
| 2026-11-15 | Owner decision: publish the trained FP32 model (`model.onnx`). This is the repo owner's decision, not a permission confirmed by the rules. | The data is 13 years old, widely used, and other teams have published models trained on it. The dataset itself is never committed or released; the repo only links to it. |
| 2026-11-15 | FER2013, FER+ labels and JAFFE are marked `status: done` in `datasets.yaml`, so the dataset table says "Used here". UIBVFEDPlus-Light, CK+ and KDEF stay `planned`. | Result files exist for the first three (`results/study3_ferplus.json`, `results/study3_ood_jaffe.json`, `results/study1_jaffe.json`). |
| 2026-11-15 | FER2013 `source_url` changed to the competition Data tab URL. | That is the page where the terms were read. |
| 2026-11-15 | Older rows in this table are kept as history and marked superseded instead of being rewritten. | Keeps the record of what was known on each date. |
| 2026-11-15 | `docs/release_notes.md` was started in Stage 1 with only the licence and attribution sections. File size, PrivateTest numbers and the INT8 caveat are added in Stage 3. | The Stage 1 brief asks for the terms note in the release notes. |
| 2026-11-15 | In `CITATION.cff`, the Goodfellow et al. (2013) reference lists only the first author's family name plus "et al.". | The full author list was not in the brief and was not checked against the paper. |
| 2026-11-15 | README notebook references point to the `.ipynb` file, so each one names a file that exists. | The old names had no extension and matched no file. |
