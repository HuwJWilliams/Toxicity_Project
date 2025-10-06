# run_vis.py
import sys
from pathlib import Path
import pandas as pd
from vis import Visualise, RESULTS_DIR, PROJ_DIR  # import from vis.py
from group_descriptors import getGroups  # if needed

# ------------------- Setup -------------------
vis = Visualise(save_all=False)
print("Visualise module loaded")

#Boxplots for cross-embedding predictions
emb_desc_path = Path("/users/yhb18174/TL_project/results/embeddings_and_descriptor_predictions")
emb_desc_dict = {
    "pred_molformer_tr_rdkit": pd.read_csv(emb_desc_path / "pred_molformer_tr_rdkit.csv", index_col=0),
    "pred_molformer_tr_mordred": pd.read_csv(emb_desc_path / "pred_molformer_tr_mordred.csv", index_col=0),
    "pred_chemberta_tr_rdkit": pd.read_csv(emb_desc_path / "pred_chemberta_tr_rdkit.csv", index_col=0),
    "pred_chemberta_tr_mordred": pd.read_csv(emb_desc_path / "pred_chemberta_tr_mordred.csv", index_col=0),
    "pred_mordred_tr_chemberta": pd.read_csv(emb_desc_path / "pred_mordred_tr_chemberta.csv", index_col=0),
    "pred_mordred_tr_molformer": pd.read_csv(emb_desc_path / "pred_mordred_tr_molformer.csv", index_col=0),
    "pred_rdkit_tr_chemberta": pd.read_csv(emb_desc_path / "pred_rdkit_tr_chemberta.csv", index_col=0),
    "pred_rdkit_tr_molformer": pd.read_csv(emb_desc_path / "pred_rdkit_tr_molformer.csv", index_col=0),
}

for key, item in emb_desc_dict.items():
    item.index.name = "Feature"

vis.plotBoxPlots(
    *emb_desc_dict.values(),
    trained_labels=["rdkit", "mordred", "rdkit", "mordred", "chemberta", "molformer", "chemberta", "molformer"],
    predicted_labels=["molformer", "molformer", "chemberta", "chemberta", "mordred", "mordred", "rdkit", "rdkit"],
    save_plot=False,
    save_path=emb_desc_path
)

#Model performance barplots
perf_files = {
    "RDKit": RESULTS_DIR / "rdkit_tr_ld50_pred" / "LD50_internal_performance_dict.json",
    "Mordred": RESULTS_DIR / "mordred_tr_ld50_pred" / "LD50_internal_performance_dict.json",
    "ChemBERTa": RESULTS_DIR / "chemberta_tr_ld50_pred" / "LD50_internal_performance_dict.json",
    "MolFormer": RESULTS_DIR / "molformer_tr_ld50_pred" / "LD50_internal_performance_dict.json",
}

vis.plotModelPerformanceBars(
    base_path=RESULTS_DIR,
    model_jsons=perf_files,
    model_labels=list(perf_files.keys()),
    metrics=["r2", "Pearson_r", "RMSE", "Bias"],
    save=False,
    save_dir=RESULTS_DIR / "performance_plots",
    plot_fsuffix="internal"
)

# Example 3: PCA of embeddings
desc = "chemberta"
save_path = RESULTS_DIR
emb_path = Path(PROJ_DIR / "datasets" / "embeddings" / f"{desc}_embeddings.csv")
ld50 = pd.read_csv(emb_path, index_col="ID")

fig, pca_df, loadings_df = vis.plotPCA(
    train=ld50.iloc[:5000],
    prediction=ld50.iloc[5000:],
    validation=None,
    source_ls=["Trained Mols", "Prediction Mols"],
    n_components=5,
    plot_area=False,
    save_plot=False,
    plot_title=f"{desc.upper()} PCA",
    save_dir=save_path,
    plot_fname=f"{desc}_pca_test"
)
