# Decisions

Defaults chosen where the project brief left a choice. Change one by editing this file and telling me.

| Date | Decision | Reason |
|---|---|---|
| 2026-11-08 | New repo from scratch, no history from the earlier repos. | The earlier repos contain files derived from restricted datasets. |
| 2026-11-08 | Code under MIT. Datasets and third-party models keep their own terms. | Standard for a portfolio repo. |
| 2026-11-08 | Study 1 primary data: UIBVFEDPlus-Light, grouped by character. | CC BY 4.0, many subjects, no faces of real people. Download is 33.7 GB, so a partial download or subset is checked in Phase 2. |
| 2026-11-08 | JAFFE is supplied by the user, used for evaluation only, never downloaded by code. | The provider forbids redistribution, including via Colab. |
| 2026-11-08 | FER2013 is downloaded by the user's own Kaggle account. Terms still to be read. | Terms unconfirmed. |
| 2026-11-08 | Grouped 5-fold with an inner group-aware validation split. | Keeps the test fold untouched. |
| 2026-11-08 | No code derived from the research-placement repo. Study 2 uses synthetic data. | The repo was not available and the real data is lab-owned student data. |
| 2026-11-08 | Private-dataset numbers come from scripts run on the owner's machine, committed as JSON with numbers only. | The datasets cannot be published. |
| 2026-11-08 | Every metric comes from a run by the owner. Until then the README says "not measured". | No claim without output. |
| 2026-11-08 | `scripts/preflight.sh` reads extra banned words from `.git/info/preflight-patterns`, not from a committed list. | A committed list of words to avoid would itself name them. |
| 2026-11-08 | CI actions pinned to major version tags (checkout v4, setup-python v5). Latest versions were not checked. | No network access when written. Check before release. |
