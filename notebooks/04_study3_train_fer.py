# %% [markdown]
# # 04. Study 3: train a small image model on FER2013 (FER+ labels)
#
# Runtime: Colab with a GPU (Runtime > Change runtime type > T4 GPU).
#
# In your own browser first:
# 1. Kaggle: open the FER2013 competition page (link in `datasets.yaml`, entry `fer2013`), accept the
#    rules, and download `fer2013.tar.gz` (a zip also works).
# 2. GitHub: open `microsoft/FERPlus`, read `LICENSE.md`, and download the repository as a zip
#    (`FERPlus-master.zip`).
# 3. Put both zip files in a Google Drive folder named `facial-expression-lab-data`.
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

# %% [markdown]
# ## Unpack both zips and find the files
# The script prints the FER+ licence file. Read it, and tell me what it says.

# %% [markdown]
# ## What is in my Drive folder and in each zip?
# Nothing is extracted yet. This only lists names and counts.

# %%
DRIVE = "/content/drive/MyDrive/facial-expression-lab-data"
# !python scripts/show_tree.py {DRIVE}

# %%
import glob
import json

DRIVE = "/content/drive/MyDrive/facial-expression-lab-data"
KAGGLE_FILE = (sorted(glob.glob(f"{DRIVE}/fer2013*.tar.gz")) + sorted(glob.glob(f"{DRIVE}/fer2013*.zip")))[0]
FERPLUS_ZIP = glob.glob(f"{DRIVE}/*ERPlus*.zip")[0]
print(KAGGLE_FILE, FERPLUS_ZIP)
# !python scripts/prepare_fer_data.py --kaggle-file "{KAGGLE_FILE}" --ferplus-zip "{FERPLUS_ZIP}" --out-dir /content/fer_data
P = json.load(open("/content/fer_data/paths.json"))
FER_CSV, FERPLUS_CSV = P["fer_csv"], P["ferplus_csv"]

# %% [markdown]
# ## What did the unpacked folders look like?

# %%
# !python scripts/show_tree.py /content/fer_data --depth 3

# %%
# !python scripts/probe_dataset.py fer {FER_CSV} {FERPLUS_CSV}

# %% [markdown]
# The probe should list `pixels` and `Usage` (maybe with spaces) in the first file, and the ten vote
# columns in the second. If something looks wrong, stop and paste the output back.

# %%
# !python scripts/study3.py train --fer-csv {FER_CSV} --ferplus-csv {FERPLUS_CSV} --epochs 15

# %%
# !mkdir -p {DRIVE}/study3 && cp data/cache/study3/model.pt data/cache/study3/meta.json {DRIVE}/study3/
# !cat data/cache/study3/meta.json | head -c 600

# %% [markdown]
# The model files are saved to your Drive. Do not commit them. Notebook 05 evaluates and exports.
