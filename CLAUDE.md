# Instructions for AI coding tools

This repo holds three facial expression studies, a small Python package (`src/fel`), thin notebooks and a browser demo.

Rules:
- Never write a claim, number or tool that the repo's own output does not show. Label every number: measured (Colab T4), measured (host CPU), measured (local machine), measured (browser), estimated, simulated, reported, or not measured.
- No datasets, images, archives or model files in git. `datasets.yaml` is the source of truth for dataset terms. Code may only download datasets marked `fetch: runtime`.
- Split by subject wherever subject IDs exist. Keep tuning and test data apart.
- Australian spelling. Plain, direct wording. No emojis.
- Run `pytest`, `ruff check .`, `python scripts/make_dataset_table.py --check`, `python scripts/make_results_table.py --check` and `python scripts/sync_notebooks.py --check` before finishing a task.
- Notebooks are paired: edit `notebooks/*.py`, then run `python scripts/sync_notebooks.py`. Never commit notebook outputs.
- Results go in `results/*.json` through `fel.results.write_result`. Never type a result into the README by hand.

## AI log

After each task, add a draft entry to `private/ai-log-drafts.md` (ignored by git) using the format at the top of `docs/ai-log.md`. Mark it `DRAFT: to be confirmed`. The owner moves reviewed entries into `docs/ai-log.md`. Record only what actually happened: what was asked, what failed on your own test or lint run, what you could not verify. Never invent a problem, a check or a result. Do not run `git commit`, `git push` or `git config`.
