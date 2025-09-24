# %%
from models.transfer_model import TL
from RFRegressor import RFRegressor
import pandas as pd
from pathlib import Path

FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[1]

RDKIT_DESC_PATH = Path(PROJ_DIR / "datasets"/"descriptors"/"rdkit_descriptors.csv")
MORDRED_DESC_PATH = Path(PROJ_DIR / "datasets"/"descriptors"/"mordred_descriptors.csv")

CHEMBERTA_EMB_PATH = Path(PROJ_DIR / "datasets"/"embeddings"/"chemberta_embeddings.csv")
MOLFORMER_EMB_PATH = Path(PROJ_DIR / "datasets"/"embeddings"/"molformer_embeddings.csv")

LD50_TARGETS_PATH = Path(PROJ_DIR / "datasets" / "LD50" / "LD50_targets_rat-mouse-oral_lt9500.csv")

MULTI_TARGET = False
# %%

rdkit_desc_df = pd.read_csv(RDKIT_DESC_PATH, index_col="ID").drop(columns=["SMILES"], errors="ignore")
mordred_desc_df = pd.read_csv(MORDRED_DESC_PATH, index_col="ID").drop(columns=["SMILES"], errors="ignore")

chemberta_emb_df  = pd.read_csv(CHEMBERTA_EMB_PATH, index_col="ID")
molformer_emb_df  = pd.read_csv(MOLFORMER_EMB_PATH, index_col="ID")
ld50_df = pd.read_csv(LD50_TARGETS_PATH, index_col="ID")['LD50']


train_data_ls = [rdkit_desc_df, mordred_desc_df, chemberta_emb_df, molformer_emb_df]
test_data_ls = [ld50_df, ld50_df, ld50_df, ld50_df]
identifier_ls =["rdkit_tr_ld50_pred", "mordred_tr_ld50_pred", "chemberta_tr_ld50_pred", "molformer_tr_ld50_pred"]

# %%

for tr, te, id in zip(train_data_ls, test_data_ls, identifier_ls):
    print(f"Shape of training df: {tr.shape}")
    print(f"Shape of testing df: {te.shape}")

    model = TL(tokeniser=None, encoder=None, log_identifier=id)

    if MULTI_TARGET:
        model.trainMultiTargetRFModels(
            features_df = tr,
            targets_df  = te,
            rf_regressor_class = RFRegressor,
            output_csv = f"{id}.csv",
            existing_performance_csv = f"{id}.csv",
            hyper_params = {
                "n_estimators": [400, 500],
                "max_features": ["sqrt"],
                "max_depth": [25, 50, 75, 100],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [2, 4, 8],
            },
            n_resamples = 10,
            test_size = 0.3,
            save_path = PROJ_DIR / "results" / "embeddings_and_descriptor_predictions",
            random_seed=model.rng()
        )
    
    else:
        data = tr.join(te, how="inner")
        model.trainSingleTargetRFModel(
            data=data,
            target_column="LD50",
            rf_regressor_class=RFRegressor,
            hyper_params={
                "n_estimators": [400, 500],
                "max_features": ["sqrt"],
                "max_depth": [25, 50, 75, 100],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [2, 4, 8],
            },
            n_resamples=10,
            test_size=0.3,
            save_path=PROJ_DIR / "results" / "LD50_predictions_rf" / id,
            random_seed=model.rng()
        )