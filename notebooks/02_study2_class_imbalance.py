# %% [markdown]
# # 02. Study 2: accuracy when one class dominates (simulated)
#
# The research placement used real student expression data that cannot be shared. This
# notebook generates data with a similar *shape*: about 92% of rows in one class, the rest small.
# The numbers are **simulated** and say nothing about the real data.
#
# Evaluation is user-grouped 5-fold. Fixes are applied to training rows only.

# %%
import pandas as pd

from fel.studies import imbalance_demo

# %%
res = imbalance_demo(seed=0)
print("share of the majority class:", round(res["majority_class_share"], 3))
print("accuracy of always predicting the majority class:",
      round(res["always_predict_majority_accuracy"], 3))
pd.DataFrame(res["table"])[["name", "accuracy", "macro_f1"]].round(3)

# %% [markdown]
# ## What to look for
# - Compare each baseline's accuracy with the always-predict-majority accuracy.
# - Compare accuracy with macro F1.
# - Fixes (class weights, oversampling) change the trade-off between the two. Read the table;
#   do not assume either improves everything.
