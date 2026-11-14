# %% [markdown]
# # 03. Study 1: landmark features with a subject-wise evaluation
#
# Run this on your own machine or in Colab. It uses UIBVFEDPlus-Light (CC BY 4.0, see
# `datasets.yaml`). The dataset is 33.7 GB, so check whether a part of it is enough first.
# Nothing in this notebook should be committed with outputs.

# %%
# !git clone https://github.com/cesinata/facial-expression-lab
# %cd facial-expression-lab
# !pip install -q -e ".[landmarks,images,plots]"

# %% [markdown]
# ## 1. Download the dataset yourself
# Follow the link in `datasets.yaml` (entry `uibvfedplus_light`), read the terms, and put the
# files in a folder. Set the path below.

# %%
DATA = "/content/uib"  # change me
# !python scripts/probe_dataset.py folder {DATA}

# %% [markdown]
# ## 2. Choose label and subject patterns
# From the example paths printed above, write two regular expressions. Each needs one capture
# group, and is matched against the path relative to the folder, with a leading slash.
# `LABEL_MAP` renames labels to FER+ names (anger, disgust, fear, happiness, sadness, surprise).

# %%
LABEL_REGEX = r"/(\w+)_"      # change me
GROUP_REGEX = r"^/(\w+)/"     # change me
LABEL_MAP = "joy=happiness"   # change me

# %%
# !python scripts/extract_features.py --source regex --root {DATA} --label-regex '{LABEL_REGEX}' --group-regex '{GROUP_REGEX}' --label-map '{LABEL_MAP}' --out data/cache/uib.npz --crops

# %% [markdown]
# ## 3. Check the selected landmark points on one image
# This only displays the figure. Do not save or commit it.

# %%
import matplotlib.pyplot as plt

from fel import landmarks as L

example_image = f"{DATA}/EXAMPLE.png"  # change me
ex = L.FaceExtractor(L.ensure_model())
lm, _, _ = ex.extract(example_image)
pts = L.all_points()
colour = {r: i for i, r in enumerate(L.REGION_POINTS)}
plt.figure(figsize=(4, 6))
plt.scatter(lm[pts, 0], -lm[pts, 1], c=[colour[L.region_of_point(p)] for p in pts], cmap="tab10")
plt.axis("equal")
plt.title("selected landmarks by region")
plt.show()

# %% [markdown]
# ## 4. Run the study
# Writes `results/study1_uibvfedplus_light.json`. Use `--models majority,logreg,mlp,fernet`
# to add the 1D CNN (needs PyTorch).

# %%
# !python scripts/run_study1.py --features data/cache/uib.npz --dataset uibvfedplus_light --label "measured (local machine)" --models majority,logreg,mlp

# %% [markdown]
# Then run `python scripts/make_results_table.py` and commit the JSON file yourself.
