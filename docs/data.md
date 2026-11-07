# Data

Terms for every dataset are stored once in `datasets.yaml`. The table below is generated from it. Do not edit the table by hand.

<!-- datasets:start -->
| Dataset | What it is | Access and terms | Licence status | In this repo |
|---|---|---|---|---|
| FER2013 | 35,887 grayscale 48x48 web images, 7 emotions, no subject IDs. | Kaggle competition data. A Kaggle account is needed and the competition rules must be accepted before download. [Provider page](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge). | unconfirmed | Planned (downloaded at run time) |
| FER+ labels | New labels for FER2013 from 10 annotators per image (label file only, no images). | Public GitHub repository (archived 13 June 2024) with a LICENSE.md file. [Provider page](https://github.com/microsoft/FERPlus). | unconfirmed | Planned (downloaded at run time) |
| JAFFE | 213 images, 10 posers, 7 expressions. | Zenodo record 14974867. Non-commercial scientific research only. Redistribution is not permitted (including via GitHub, Kaggle and Colab). Images must not be posted on the web. [Provider page](https://zenodo.org/records/14974867). | confirmed | Planned (you download it yourself) |
| UIBVFEDPlus-Light | 3,300 PNG images (1080x1920) of 100 virtual characters under 4 lighting setups, six emotion labels. | Download from the University of the Balearic Islands project page (33.7 GB). [Provider page](https://ugivia.uib.es/?p=1545). | partly confirmed | Planned (downloaded at run time) |
| CK+ (Extended Cohn-Kanade) | 593 video sequences, 123 subjects, lab-posed expressions. | Request from the authors. Non-commercial. A consent form limits publishing faces. [Provider page](https://www.jeffcohn.net/resources/). | partly confirmed | Private results only (planned) |
| KDEF | 4,900 pictures, 70 actors, 7 expressions, 5 angles (figures from the project brief; not checked). | Non-commercial research use. Request through the provider. [add provider link] | unconfirmed | Private results only (planned) |
| HSS Learning | Student expression logs from an e-learning platform, collected by a university lab. | Not public (student data). | partly confirmed | Private results only (planned) |
| AffectNet | Large in-the-wild expression dataset. | Request to the authors, research use. [add provider link] | unconfirmed | Not used |
| RAF-DB | Large in-the-wild expression dataset. | Request to the authors, research use. [add provider link] | unconfirmed | Not used |
| AFEW | Film clips with expression labels. | Academic, non-commercial licence; no redistribution. [add provider link] | unconfirmed | Not used |

Licence information last checked: 2026-11-08. Entries marked unconfirmed have not been read on the provider's page.
<!-- datasets:end -->

## Rules

1. Never commit raw datasets, images, archives or model weights. `.gitignore` and `scripts/preflight.sh` block them.
2. Read the provider's own page before using a dataset. Record what you read and the date in `datasets.yaml`.
3. Private data stays on your machine. `data/private/` is ignored by git except for its README. A folder name does not make anything private on GitHub, so do not rely on the name.
4. `fel.data.download(name)` prints the licence and refuses anything the registry marks `public: false` or `fetch: never`.

## Private data layout (on your machine only)

Pass the folder with a command-line flag, for example `--ckplus-dir`. The folders can live anywhere. `data/private/ckplus/` and `data/private/kdef/` are convenience locations.

CK+ layout expected by the thesis-era code: `extended-ck/<subject>/<sequence>/<frame>.png`, for example `S005/001/S005_001_00000001.png`, plus a label file. Phase 2 will document the exact label format and use only sequences with provider labels.

KDEF: the standard folder-per-actor layout. The expected structure is confirmed in Phase 2 when the loader is written.

## Notes recorded so far

- JAFFE: the provider's page limits use to non-commercial scientific research and forbids redistribution, including via Colab. This repo never downloads it and never shows its images.
- UIBVFEDPlus-Light: CC BY 4.0 on the project page and the university repository record (read 2026-11-08). Confirm on the figshare record before release.
- FER+ and FER2013: licence text not read yet. Not confirmed.
- CK+ labels: Phase 2 uses only sequences that carry provider labels and reports any hand-labelled sequences separately.
