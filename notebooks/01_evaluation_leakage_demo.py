# %% [markdown]
# # 01. Why a random split of frames inflates results (simulated)
#
# Everything here is **simulated**: 30 subjects, 4 classes, 4 near-identical frames per
# sequence (like the first and last frames of a video clip). No dataset is used.
#
# Three protocols are compared on the same data and the same models:
# - **legacy**: select features on all rows, then a random 70/15/15 split of rows (the thesis protocol)
# - **random 5-fold**: split rows at random
# - **subject-wise 5-fold**: no subject appears in both train and test

# %%
import pandas as pd

from fel.studies import leakage_demo, selection_leak_demo

# %%
leak = leakage_demo(seeds=range(5))
pd.DataFrame(leak["table"])[["name", "accuracy", "accuracy_std", "macro_f1"]].round(3)

# %% [markdown]
# ## How to read it
# Chance accuracy is 0.25. A model that can match a test frame to a sibling frame in the
# training set (1-nearest-neighbour) looks excellent under the first two protocols and drops
# under the subject-wise one. Linear models are affected less. Whatever the table above shows
# is what this run measured.
#
# ## Feature selection on all rows
# Features here are pure noise, so any accuracy above chance (0.25) is leakage.

# %%
sel = selection_leak_demo(seeds=range(5))
pd.DataFrame(sel["table"])[["name", "accuracy", "accuracy_std"]].round(3)

# %% [markdown]
# Report the size of the gap as measured. In the thesis code, Mutual Information was computed
# on all rows before the split. This table shows how much that alone can matter in a simulated
# setting. The near-duplicate frames are a separate problem.
