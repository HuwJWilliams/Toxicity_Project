"""Script to train a Multitarget Random Forest Regressors i.e., training to predict each column in target_df"""

import sys
import pandas as pd
from pathlib import Path
from models.transfer_model import TL

PALMERCHEM_SOFTWARE = Path.home() / "PalmerChem_Software" / "src" / "models"
sys.path.insert(0, str(PALMERCHEM_SOFTWARE))
from RFRegressor import RFRegressor

# --- Arguments
TRAIN_NAME = sys.argv[1]
TEST_NAME = sys.argv[2]
IDENTIFIER = sys.argv[3]

print(TRAIN_NAME)
print(TEST_NAME)
print(IDENTIFIER)

# --- Paths
FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[1]

RDKIT_DESC_PATH = Path(PROJ_DIR / "datasets"/ "descriptors" / "rdkit_descriptors.csv")
MORDRED_DESC_PATH = Path(PROJ_DIR / "datasets" / "descriptors" / "mordred_descriptors.csv")
CHEMBERTA_EMB_PATH = Path(PROJ_DIR / "datasets"/ "embeddings" / "chemberta_embeddings.csv")
MOLFORMER_EMB_PATH = Path(PROJ_DIR / "datasets"/ "embeddings" / "molformer_embeddings.csv")
LD50_TARGETS_PATH = Path(PROJ_DIR / "datasets" / "LD50" / "LD50_targets_rat-mouse-oral_lt9500.csv")

# --- Load datasets
datasets = {
    "rdkit":     pd.read_csv(RDKIT_DESC_PATH, index_col="ID").drop(columns=["SMILES"], errors="ignore").head(1000),
    "mordred":   pd.read_csv(MORDRED_DESC_PATH, index_col="ID").drop(columns=["SMILES"], errors="ignore").head(1000),
    "chemberta": pd.read_csv(CHEMBERTA_EMB_PATH, index_col="ID").head(1000),
    "molformer": pd.read_csv(MOLFORMER_EMB_PATH, index_col="ID").head(1000),
}

train_df = datasets[TRAIN_NAME]
target_df = datasets[TEST_NAME]

common_idx = train_df.index.intersection(target_df.index)

train_df = train_df.loc[common_idx]
target_df = target_df.loc[common_idx]

# --- Train model
print(f"[Multi-Target RF] train={TRAIN_NAME}, test={TEST_NAME}, id={IDENTIFIER}")
print(f"Train shape: {train_df.shape}, Test shape: {target_df.shape}")

model=TL(tokeniser=None, encoder=None, log_identifier=IDENTIFIER)
model.trainMultiTargetRFModels(
    features_df=train_df,
    targets_df=target_df,
    rf_regressor_class=RFRegressor,
    output_csv=f"{IDENTIFIER}.csv",
    existing_performance_csv=f"{IDENTIFIER}.csv",
    hyper_params={
    "n_estimators": [100, 200],
    "max_features": ["sqrt"],
    "max_depth": [25, 50],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [2, 4, 8],
    },
    n_resamples=10,
    test_size=0.3,
    save_path=PROJ_DIR / "results" / "embeddings_and_descriptor_predictions",
    random_seed=model.rng()
)