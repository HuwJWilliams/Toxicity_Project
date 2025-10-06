"""Script to predict with a single Random Forest Regressor and save outputs"""
# %%
import sys
import pandas as pd
from pathlib import Path
from models.transfer_model import TL

PALMERCHEM_SOFTWARE = Path.home() / "PalmerChem_Software" / "src" / "models"
sys.path.insert(0, str(PALMERCHEM_SOFTWARE))
from RFRegressor import RFRegressor

# --- Arguments
# TRAIN_NAME = sys.argv[1]
# print(TRAIN_NAME)

# TEST_NAME = sys.argv[2]
# print(TEST_NAME)

IDENTIFIER = "rdkit_test"
print(IDENTIFIER)

TARGET_COLUMN = "LD50"
print(TARGET_COLUMN)

# --- Paths
FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[1]

RDKIT_DESC_PATH = Path(PROJ_DIR / "datasets"/"descriptors"/"rdkit_descriptors.csv")
MORDRED_DESC_PATH = Path(PROJ_DIR / "datasets"/"descriptors"/"mordred_descriptors.csv")
CHEMBERTA_EMB_PATH = Path(PROJ_DIR / "datasets"/"embeddings"/"chemberta_embeddings.csv")
MOLFORMER_EMB_PATH = Path(PROJ_DIR / "datasets"/"embeddings"/"molformer_embeddings.csv")
LD50_TARGETS_PATH = Path(PROJ_DIR / "datasets" / "LD50" / "LD50_targets_rat-mouse-oral_lt9500.csv")

# --- Load datasets
datasets = {
    "rdkit":     pd.read_csv(RDKIT_DESC_PATH, index_col="ID").drop(columns=["SMILES"], errors="ignore").iloc[5000:],
    "mordred":   pd.read_csv(MORDRED_DESC_PATH, index_col="ID").drop(columns=["SMILES"], errors="ignore").iloc[5000:],
    "chemberta": pd.read_csv(CHEMBERTA_EMB_PATH, index_col="ID").iloc[5000:],
    "molformer": pd.read_csv(MOLFORMER_EMB_PATH, index_col="ID").iloc[5000:],
    "ld50":      pd.read_csv(LD50_TARGETS_PATH, index_col="ID").iloc[5000:]
}

data = "molformer"
feats_pred = datasets[data]
targs_true = datasets["ld50"]
# %%

common_idx = feats_pred.index.intersection(targs_true.index)

feats_pred = feats_pred.loc[common_idx]
targs_true = targs_true.loc[common_idx]

path = Path(f"/users/yhb18174/TL_project/results/LD50_predictions_rf/{data}_tr_ld50_pred")


model = TL(tokeniser=None, encoder=None, log_identifier=IDENTIFIER)
pred, true, perf_dict = model.predictSingleTargetRF(
    model= path / "LD50_RF_model.pkl",
    feature_data=feats_pred,
    calc_perf=True,
    targets_true=targs_true,
    save_preds=True,
    save_path=path,
    preds_filename="LD50_predictions.csv",
    perf_filename="LD50_prediction_performance.json"
    )
# %%

import matplotlib.pyplot as plt

# Example data
x = pred
y = true

plt.figure(figsize=(6,4))
plt.scatter(x, y, color="blue", alpha=0.7, edgecolors="k")
plt.xlabel("Pred values")
plt.ylabel("True values")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# %%

