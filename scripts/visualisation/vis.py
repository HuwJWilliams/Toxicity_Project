# %%
import pandas as pd
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, "../")
from group_descriptors import getGroups

# --- Paths
FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[2]
RESULTS_DIR = Path(PROJ_DIR / "results" / "LD50_predictions_rf")

# --- Load data
working_dir = RESULTS_DIR / "rdkit_tr_ld50_pred"
data = "LD50_feature_importance.csv"

x = pd.read_csv(working_dir / data)
x = x.sort_values(by="Importance", ascending=False)

# %%
class Visualise():
    def __init__(
            self,
            save_all: bool=False,
            save_path: Path=None
    ):
        self.save_all = save_all
        self.save_path = save_path

    def plotFeatureImportance(
            self,
            data: pd.DataFrame,
            title: str=None,
            x_col: str="Importance",
            y_col: str="Feature",
            title_fontsize: int=16,
            label_fontsize: int=14,
            tick_fontsize: int=12,
            top_n: int=None,
            figsize: tuple=(10, 8),
            save_plot: bool=False,
            save_path: Path=None,
            save_fname: str="feature_importance.png",
            dpi: int=400
    ):        
        if top_n:
            data = data.head(top_n)

        palette = sns.color_palette("tab10", n_colors=len(data))
        color_map = dict(zip(data["Feature"], palette))


        plt.figure(figsize=figsize)
        sns.barplot(
            data=data,
            x=x_col,
            y=y_col,
            palette=color_map,
            dodge=False,
            hue=y_col,
            legend=False
        )

        plt.title(title, fontsize=title_fontsize)
        plt.ylabel(y_col.capitalize(), fontsize=label_fontsize)
        plt.xlabel(x_col.capitalize(), fontsize=label_fontsize)

        plt.xticks(fontsize=tick_fontsize, rotation=45)
        plt.yticks(fontsize=tick_fontsize)
        plt.tight_layout()

        if save_plot or self.save_all:
            save_path = save_path or self.save_path
            plt.savefig(save_path / save_fname, dpi=dpi)


    def plotMultiTaskPerformance(
            self,
            data: pd.DataFrame,
            x_col: str,
            y_col: str,
            ascending: bool,
            top_n: int,
            subset: str=None,
            figsize: tuple=(10, 8),
            title: str="Multi-Task Performance",
            title_fontsize: int=16,
            label_fontsize: int=14,
            tick_fontsize: int=12,
            save_plot: bool=False,
            save_path: Path=None,
            save_fname: str="multi-task_performance.png",
            dpi: int=400
    ):
        
        if subset:
            data = data[data.index.astype(str).str.contains(subset, case=False, na=False)]

        data = data.copy()
        data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
        data = data.dropna(subset=[x_col])
        data = data.sort_values(by=x_col, ascending=ascending).head(top_n)

        plt.figure(figsize=figsize)
        plt.barh(data.index, data[x_col].values)
        plt.title(title, fontsize=title_fontsize)

        plt.xlabel(x_col.capitalize(), fontsize=label_fontsize)
        plt.ylabel(y_col.capitalize(), fontsize=label_fontsize)

        plt.yticks(fontsize=tick_fontsize)
        plt.xticks(fontsize=tick_fontsize)

        plt.tight_layout()

        if save_plot or self.save_all:
            save_path = save_path or self.save_path
            plt.savefig(save_path / save_fname, dpi=dpi)

        plt.show()

    def computeGroupPerf(
            self,
            data: pd.DataFrame,
            descriptor_groups: dict,
            metric="Pearson_r"
        ) -> pd.DataFrame:
        """Compute average performance per descriptor group."""
        group_perf = {}
        for group, descs in descriptor_groups.items():
            valid = [d for d in descs if d in data.index]
            if valid:
                group_perf[group] = data.loc[valid, metric].mean()
        return pd.DataFrame.from_dict(group_perf, orient="index", columns=[f"avg_{metric}"])


    def plotGroupRadar(
            self, 
            group_perf_df: pd.DataFrame,
            title="Radar Plot",
            fontsize=12,
            title_size=16,
            savepath=None
        ):
        """Make a radar plot of group performances with dual scales (0–1 and 0.5–1)."""
        labels = group_perf_df.index.tolist()
        values = group_perf_df.iloc[:,0].values
        values = np.append(values, values[0])
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        angles = np.append(angles, angles[0])

        palette = sns.color_palette("tab10")
        blue, red = palette[0], palette[3]

        fig = plt.figure(figsize=(10,10))

        # Axis 1 (0–1)
        ax1 = fig.add_subplot(111, polar=True)
        ax1.plot(angles, values, linewidth=2, color=blue, label="0–1 scale", alpha=0.5, zorder=1)
        ax1.fill(angles, values, alpha=0.2, color=blue, zorder=0)

        ax1.set_ylim(0,1)
        ticks_blue = np.arange(0,1.1,0.1)
        ax1.set_yticks(ticks_blue)
        ax1.set_yticklabels([f"{t:.1f}" if t>0 else "" for t in ticks_blue], 
                            color=blue, fontsize=fontsize, fontweight="bold")
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(labels, fontsize=fontsize, fontweight="bold")
        ax1.set_rlabel_position(90)

        # Axis 2 (0.5–1)
        ax2 = fig.add_subplot(111, polar=True, frame_on=False)
        ax2.plot(angles, values, linewidth=2, color=red, linestyle="--", label="0.5–1 scale", alpha=0.5, zorder=1)
        ax2.fill(angles, values, alpha=0.15, color=red, zorder=0)

        ax2.set_ylim(0.5,1)
        ticks_red = np.arange(0.5,1.01,0.1)
        ax2.set_yticks(ticks_red)
        ax2.set_yticklabels([f"{t:.2f}" for t in ticks_red], 
                            color=red, fontsize=fontsize, fontweight="bold")
        ax2.set_xticks([])
        ax2.set_xticklabels([])
        ax2.set_rlabel_position(270)

        # Merge legends
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1+handles2, labels1+labels2, loc="upper right", bbox_to_anchor=(1.2,1.1), fontsize=fontsize)

        plt.title(title, fontsize=title_size, weight="bold", pad=40)

        if savepath:
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.show()


    def plotGroupBar(
            self, 
            *group_dfs: pd.DataFrame,
            labels: list[str] = None,
            title: str = "Group Comparison",
            y_label: str = "Average Pearson r",
            x_label: str = "Descriptor Group",
            palette: str | list = "tab10",
            figsize: tuple = (12,6),
            rotation: int = 45,
            fontsize=12,
            title_size=16,
            savepath=None
        ):
        """Grouped barplot for any number of models' performances."""
        if labels is None:
            labels = [f"Model{i+1}" for i in range(len(group_dfs))]
        if len(labels) != len(group_dfs):
            raise ValueError("Number of labels must match number of DataFrames")

        dfs_long = []
        for df, label in zip(group_dfs, labels):
            temp = df.copy()
            temp["dataset"] = label
            temp["group"] = temp.index
            dfs_long.append(temp)
        all_df = pd.concat(dfs_long)

        plt.figure(figsize=figsize)
        sns.barplot(
            data=all_df,
            x="group",
            y=all_df.columns[0],
            hue="dataset",
            palette=palette
        )
        plt.xticks(rotation=rotation, ha="right", fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylabel(y_label, fontsize=fontsize, weight="bold")
        plt.xlabel(x_label, fontsize=fontsize, weight="bold")
        plt.title(title, fontsize=title_size, weight="bold")
        plt.legend(fontsize=fontsize)
        plt.tight_layout()

        if savepath:
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.show()


    def plotGroupHeatmaps(
            self,
            data: pd.DataFrame,
            descriptor_groups: dict,
            metric: str = "Pearson_r",
            title_prefix: str = "Descriptor Performance",
            cmap: str = "viridis",
            annot: bool = True,
            vmin: float = 0.0,
            vmax: float = 1.0,
            fontsize=12,
            title_size=14,
            savepath=None
        ):
        """Plot one heatmap per descriptor group."""
        for group, descs in descriptor_groups.items():
            valid = [d for d in descs if d in data.index]
            if not valid:
                continue

            df_group = data.loc[valid, [metric]].sort_values(metric, ascending=False)

            plt.figure(figsize=(6, max(4, len(valid) * 0.25)))
            sns.heatmap(df_group,
                        cmap=cmap,
                        annot=annot,
                        fmt=".2f",
                        cbar_kws={"label": metric},
                        vmin=vmin,
                        vmax=vmax,
                        annot_kws={"fontsize": fontsize})
            plt.title(f"{title_prefix}: {group}", fontsize=title_size, weight="bold", pad=20)
            plt.ylabel("Descriptor", fontsize=fontsize, weight="bold")
            plt.xlabel("")
            plt.yticks(fontsize=fontsize)
            plt.tight_layout()

            if savepath:
                outpath = f"{savepath.rstrip('.png')}_{group}.png"
                plt.savefig(outpath, dpi=300, bbox_inches="tight")
            plt.show()


# %%
        
data1 = pd.read_csv("/users/yhb18174/TL_project/results/embeddings_and_descriptor_predictions/pred_rdkit_tr_molformer.csv", index_col=0)
data1.index.name = "Embedding"
vis=Visualise()
# vis.plotMultiTaskPerformance(data=data1,
#                              x_col="Pearson_r",
#                              y_col="Embedding",
#                              ascending=False, 
#                              top_n=20)
# %%
data2 = pd.read_csv("/users/yhb18174/TL_project/results/embeddings_and_descriptor_predictions/pred_rdkit_tr_chemberta.csv", index_col=0)
data2.index.name = "Embedding"
vis=Visualise()
# vis.plotMultiTaskPerformance(data=data2,
#                              x_col="Pearson_r",
#                              y_col="Embedding",
#                              ascending=False, 
#                              top_n=20)

# Compute per-group averages
perf1 = vis.computeGroupPerf(data1, getGroups("rdkit"))
perf2 = vis.computeGroupPerf(data2, getGroups("rdkit"))

SAVE_PATH = Path(PROJ_DIR / "results" / "embeddings_and_descriptor_predictions" / "rdkit_pred_vis")
# Radar for MolFormer
vis.plotGroupRadar(perf1, title="MolFormer Group Performance", savepath=SAVE_PATH / "molformer_radar.png")
vis.plotGroupRadar(perf2, title="MolFormer Group Performance", savepath=SAVE_PATH / "chemberta_radar.png")

# # Compare barplot
vis.plotGroupBar(perf1, perf2, labels=["MolFormer", "ChemBERTa"], savepath=SAVE_PATH / "chemberta_vs_molformer_barchart.png")

# # Heatmaps for ChemBERTa
vis.plotGroupHeatmaps(data1, getGroups("rdkit"), savepath=SAVE_PATH / "molformer_heatmap.png")

vis.plotGroupHeatmaps(data2, getGroups("rdkit")savepath=SAVE_PATH / "chemberta_heatmap.png")

# %%
