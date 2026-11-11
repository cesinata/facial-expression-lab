# %% [markdown]
# # 04. Study 3: train a small image model on FER2013 (FER+ labels)
#
# Runtime: Colab with a GPU (Runtime > Change runtime type > T4 GPU).
#
# Do this in your own browser first:
# 1. Kaggle: open the FER2013 competition page (link in `datasets.yaml`, entry `fer2013`), read and
#    accept the rules, and download `fer2013.tar.gz` from the Data tab.
# 2. GitHub: open `microsoft/FERPlus`, read `LICENSE.md`, and download `fer2013new.csv`.
# 3. Put both files in a Google Drive folder named `facial-expression-lab-data`.
#
# This notebook trains on the Training split and chooses the checkpoint on PublicTest.
# It never touches PrivateTest.

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
# !mkdir -p /content/fer_data && tar -xzf {DRIVE}/fer2013.tar.gz -C /content/fer_data
FER_CSV = glob.glob("/content/fer_data/**/fer2013.csv", recursive=True)[0]
FERPLUS_CSV = f"{DRIVE}/fer2013new.csv"
print(FER_CSV, FERPLUS_CSV)
# !python scripts/probe_dataset.py fer {FER_CSV} {FERPLUS_CSV}

# %% [markdown]
# The probe should list `pixels` and `Usage` in the first file and ten vote columns in the second.
# If the names differ, stop and paste the output back.

# %%
# !python scripts/study3.py train --fer-csv {FER_CSV} --ferplus-csv {FERPLUS_CSV} --epochs 15

# %%
# !mkdir -p {DRIVE}/study3 && cp data/cache/study3/model.pt data/cache/study3/meta.json {DRIVE}/study3/
# !cat data/cache/study3/meta.json | head -c 600

# %% [markdown]
# The model files are saved to your Drive. Do not commit them. Notebook 05 evaluates and exports.
