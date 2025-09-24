from models.transfer_model import TL
from RFRegressor import RFRegressor
import pandas as pd
from pathlib import Path

FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[1]

DESC_PATH = Path(PROJ_DIR / "datasets"/"descriptors"/"rdkit_descriptors.csv")
EMB_PATH = Path(PROJ_DIR / "datasets"/"embeddings"/"chemberta_embeddings.csv")

desc_df = pd.read_csv(DESC_PATH, index_col="ID").drop(columns=["SMILES"], errors="ignore").head(1000)
emb_df  = pd.read_csv(EMB_PATH, index_col="ID").head(2000)

identifier = "pred_chemberta_tr_rdkit"

model = TL(tokeniser=None, encoder=None, log_identifier=identifier)

model.trainMultiTargetRFModels(
    features_df = desc_df,
    targets_df  = emb_df,
    rf_regressor_class = RFRegressor,
    output_csv = f"{identifier}.csv",
    existing_performance_csv = f"{identifier}.csv",
    hyper_params = {
        "n_estimators": [400, 500],
        "max_features": ["sqrt"],
        "max_depth": [25, 50, 75, 100],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [2, 4, 8],
    },
    n_resamples = 10,
    test_size = 0.3,
    n_jobs = 20,
    save_path = PROJ_DIR / "results" / "embeddings_and_descriptor_predictions"
)


