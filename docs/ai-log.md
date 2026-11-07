# AI log

This log records AI-assisted work in this repo: what I asked, what the tool got wrong, how I checked it and what I changed. Entries are written or confirmed by me. Numbers in this repo come from runs I did, and each is labelled in the README.

Format for each entry (keep it under ten lines):

## YYYY-MM-DD: [task in one line]
- **Tool and mode:** [chat / coding agent / inline completion]
- **Spec or test written first:** [yes, `path/to/test` / no, because ...]
- **What I asked:** [one or two sentences, not the full prompt]
- **What the tool produced:** [files or functions touched]
- **What was wrong or risky:** [specific problem, or "nothing found" plus how I looked]
- **How I checked:** [commands run, tests, manual check, comparison with docs]
- **What I changed:** [edits after my review]
- **Result:** [outcome; label any number as measured / estimated / simulated]

---

## 2026-11-08: Audit of my 2023 thesis code before rebuilding it
DRAFT: to be confirmed
- **Tool and mode:** chat
- **Spec or test written first:** no. Read-only audit of my notebooks, reports and slides.
- **What I asked:** find the exact train/validation/test split and whether the reported 91.95% is a test or a validation result.
- **What the tool produced:** a list of findings with notebook, cell and line references (split is 70/15/15 at frame level; 91.95% is a test accuracy; the test set was also used to pick the feature set and hyperparameters; Mutual Information was computed before the split).
- **What was wrong or risky:** the tool flagged that class weights printed in one tuning notebook are in a different order from the other notebook, which suggests weights assigned to the wrong classes. It checked the ordering behaviour on toy labels only. It did not re-run the notebook, so the effect on accuracy is not known.
- **How I checked:** [to be filled by me after re-reading the cited cells]
- **What I changed:** nothing in the old code. The findings are recorded in `docs/evaluation.md`.
- **Result:** no new number. The old numbers stay labelled reported (thesis).

## 2026-11-08: Dataset terms check
DRAFT: to be confirmed
- **Tool and mode:** chat with web search
- **Spec or test written first:** no.
- **What I asked:** confirm the licences listed in the project brief.
- **What the tool produced:** corrections to `datasets.yaml`.
- **What was wrong or risky:** the brief listed JAFFE as CC BY 4.0 (marked unverified). The provider's own page says non-commercial scientific research only and forbids redistribution, including via Colab. One third-party mirror showed CC BY 4.0, which the tool did not use. The FER2013 and FER+ licence texts could not be read, so they stay "unconfirmed".
- **How I checked:** [to be filled by me: open the Zenodo record and the other provider pages and compare]
- **What I changed:** JAFFE is user-supplied only and never downloaded by code.
- **Result:** registry entries and their `last_checked` dates.

## 2026-11-08: Phase 1 evaluation harness (registry, splits, metrics, features, results writer)
DRAFT: to be confirmed
- **Tool and mode:** chat, code written in a sandbox without network access
- **Spec or test written first:** tests written in the same session as the code, not strictly before it (`tests/`).
- **What I asked:** subject-wise splits, macro F1 and per-class recall, a dataset registry with a download guard, and tests on synthetic data.
- **What the tool produced:** `src/fel/` modules, `scripts/`, `tests/`, CI workflow, docs.
- **What was wrong or risky:** `random_frame_split` computed the test size as `1 - 0.7`, which is 0.30000000000000004 in floating point, so scikit-learn produced 697 test rows instead of 696. `test_random_frame_split_sizes_and_disjoint` caught it. The tool rounded the fractions and then confirmed the output is identical to the two `train_test_split` calls in the thesis code. The sandbox could not install pytest or ruff, so the tests ran under a small stand-in runner and lint was not run.
- **How I checked:** [to be filled by me: run `pytest`, `ruff check .` and the two scripts in a clean virtual environment]
- **What I changed:** [to be filled by me]
- **Result:** 47 tests passed under the stand-in runner (simulated data only). Real pytest and ruff results: not measured.

## 2026-11-08: First real run of pytest and ruff, repo hygiene check
DRAFT: to be confirmed
- **Tool and mode:** coding agent (Claude Code in VS Code), local machine
- **Spec or test written first:** no. No new code or tests; fix limited to one lint rule.
- **What I asked:** run pytest, ruff, the dataset table check and the registry check, fix real failures minimally, then check git hygiene.
- **What the tool produced:** a one-line edit in `src/fel/splits.py`, `.claude/settings.json` (deny rules, excluded from git) and this entry. It also copied `.gitignore` and `.github/` from `~/Downloads/facial-expression-lab` into the repo.
- **What was wrong or risky:** `ruff check .` raised UP028 (`yield` over `for` loop) at `src/fel/splits.py:80`. Ruff marks the fix unsafe, so the tool made the edit by hand to `yield from`; the generator is only iterated, so behaviour should not change. `.gitignore` and `.github/` were missing from the repo (hidden files not copied), so `data/private/` was not ignored until they were added. All other checks passed first time. Not verified: the CI workflow itself (not run), the pinned action versions, and the `.claude/settings.json` deny syntax (`/permissions` cannot be run by the tool).
- **How I checked:** [to be filled by me]
- **What I changed:** [to be filled by me]
- **Result:** measured (local machine), Python 3.14.7, pytest 9.2.0, ruff 0.17.4, numpy 2.5.4, scikit-learn 1.9.1, pandas 3.1.1: 47 passed, ruff clean after the fix, table check and registry check passed. The CI matrix in `.github/workflows/ci.yml` is Python 3.10 and 3.12, so this run on 3.14 does not cover those versions.
