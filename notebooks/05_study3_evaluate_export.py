# %% [markdown]
# # 05. Study 3: evaluate once, calibrate, export to ONNX, test out of distribution
#
# `evaluate` reads the PrivateTest split once and refuses to run again without `--allow-repeat`.
# It runs on the CPU of this Colab machine, so the label is "measured (host CPU)", and the latency
# it records is for this machine only.
#
# Use a fresh Colab session or the same one as notebook 04. Training must be finished first.

# %%
from google.colab import drive

drive.mount("/content/drive")

# %%
# !git clone https://github.com/cesinata/facial-expression-lab
# %cd facial-expression-lab
# !pip install -q -e ".[torch,images,plots]"

# %%
import glob

DRIVE = "/content/drive/MyDrive/facial-expression-lab-data"
# !mkdir -p /content/fer_data data/cache/study3 && tar -xzf {DRIVE}/fer2013.tar.gz -C /content/fer_data
# !cp {DRIVE}/study3/model.pt {DRIVE}/study3/meta.json data/cache/study3/
FER_CSV = glob.glob("/content/fer_data/**/fer2013.csv", recursive=True)[0]
FERPLUS_CSV = f"{DRIVE}/fer2013new.csv"

# %%
# !python scripts/study3.py evaluate --fer-csv {FER_CSV} --ferplus-csv {FERPLUS_CSV} --label "measured (host CPU)"

# %%
# !mkdir -p {DRIVE}/out && cp results/study3_ferplus.json {DRIVE}/out/ && cp data/cache/study3/model.onnx {DRIVE}/out/
# !cp data/cache/study3/model_int8.onnx {DRIVE}/out/ 2>/dev/null; ls -la {DRIVE}/out

# %% [markdown]
# ## Out of distribution (optional)
# On your own computer, run `extract_features.py --source jaffe ... --crops --out data/cache/jaffe.npz`
# (see `docs/reproduce.md`), and upload `jaffe.npz` to the Drive folder. JAFFE stays in your own Drive and
# is never published. Then:

# %%
# !cp {DRIVE}/jaffe.npz data/cache/jaffe.npz
# !python scripts/study3.py ood --npz data/cache/jaffe.npz --dataset jaffe --label "measured (host CPU)"
# !cp results/study3_ood_jaffe.json {DRIVE}/out/

# %% [markdown]
# Download `study3_ferplus.json` (and `study3_ood_jaffe.json`) from `{DRIVE}/out`, put them in the
# repo's `results/` folder, and download `model.onnx` for the release. Then follow `COMMIT_PLAN.md`.
