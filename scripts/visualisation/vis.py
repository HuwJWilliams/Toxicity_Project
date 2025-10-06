"""
Script that holds the functions used to visualise all of the data in this project
"""
# %% --- Imports
import pandas as pd
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import json
from glob import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial import ConvexHull
import random as rand
from typing import Union

sys.path.insert(0, "../")
from group_descriptors import getGroups

# %% --- Pathing
FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[2]
RESULTS_DIR = Path(PROJ_DIR / "results" / "LD50_predictions_rf")

# %% --- Visualisation Class

class Visualise():
    """
    Class to hold all of the visualisation code used for this project
    """
    def __init__(
            self,
            save_all: bool=False,
            save_path: Path=None
    ):
        """
        Initialisation of the class

        Parameters
        ----------
        save_all: bool
                        Flag to be able to save all plots in the class without having to explicitly
                        define for each function.
        save_path: Path
                        Path to save the plots to

        """
        self.save_all = save_all
        self.save_path = save_path

        # Set colour consistent colour mapping for each descriptor group
        palette = plt.get_cmap("tab10")

        self.colour_map = {
            "rdkit": palette(0),
            "mordred": palette(1),
            "chemberta": palette(2),
            "molformer": palette(3)
        }

        # Set default colour to black incase labels dont match colour map
        self.default_colour = (0.0, 0.0, 0.0, 1.0)

    def _getColour(self, name:str) -> tuple:
        """
        Function to get the colour of a label
        
        Parameters
        ----------
        name: str
                    Label associated to the colour map

        Returns
        -------
        An RGBA tuple (R, G, B, A)
        """

        for key, colour in self.colour_map.items():
            if key.lower() in name.lower():
                return colour
        return self.default_colour

    def _savePlot(
            self,
            save_plot: bool, 
            save_path: Union[str, Path], 
            save_fname: str, 
            dpi: int,
            save_message: str="Saved plot"
            ):
        
        """
        Save plots

        Parameters
        ----------
        save_plot: bool
                            Flag to save the plot
        save_path: str, Path
                            Path to save the path to
        save_fname: str
                            Name to save the plot under. Must end in image extension (.png)
        dpi: int
                            Dots per inch, or quality, of saved image
        save_message: str  (optional)
                            Message to print when saving plot

        """

        if save_plot or self.save_all:
            save_path = save_path or self.save_path
            full_save_path = save_path / save_fname
            plt.savefig(full_save_path, dpi=dpi)
            print(f"{save_message}\n{full_save_path}")

    def plotFeatureImportance(
            self,
            data: pd.DataFrame,
            x_col: str="Importance",
            y_col: str="Feature",
            ascending=False,
            top_n: int=None,
            title: str=None,
            title_fontsize: int=16,
            label_fontsize: int=14,
            tick_fontsize: int=12,
            figsize: tuple=(10, 8),
            save_plot: bool=False,
            save_path: Union[Path, str]=None,
            save_fname: str="feature_importance.png",
            dpi: int=400,
    ):
        """
        Function to plot a models feature importance
        
        Parameters
        ----------
        data: pd.DataFrame
                            Dataframe containing feature importance data.
        x_col: str (optional)
                            The column name containing the numerical feature importance
                            values
        y_col: str (optional)
                            The column name containing the name of the features
        ascending: bool (optional)
                            Order to sort the data. True = low to high, False = high to low
        top_n: int (optional)
                            Changes the number of features plotted
        title: str (optional)
                            Sets the title of the plot
        title_fontsize: int (optional)
                            Changes the font size for the title
        label_fontsize: int (optional)
                            Changes the font size for the axis labels
        tick_fontsize: int (optional)
                            Changes the font size for the axis ticks
        figsize: tuple (optional)
                            Changes the size of the figure
        save_plot: bool (optional)
                            Flag to save the plot
        save_path: str, Path (optional)
                            Path to save the path to
        save_fname: str (optional)
                            Name to save the plot under. Must end in image extension (.png)
        dpi: int (optional)
                            Dots per inch, or quality, of saved image
        """

        # Sorting values by the x-column
        data = data.sort_values(by=x_col, ascending=ascending)

        # If specified, look at top_n number of features
        if top_n:
            data = data.head(top_n)

        # Setting the colour map
        palette = sns.color_palette("tab10", n_colors=len(data))
        colour_map = dict(zip(data[y_col], palette))

        # Plotting the figure
        plt.figure(figsize=figsize)
        sns.barplot(
            data=data,
            x=x_col,
            y=y_col,
            palette=colour_map,
            dodge=False,
            hue=y_col,
            legend=False
        )

        plt.title(title, fontsize=title_fontsize)
        plt.ylabel(y_col, fontsize=label_fontsize)
        plt.xlabel(x_col.capitalize(), fontsize=label_fontsize)

        plt.xticks(fontsize=tick_fontsize, rotation=45)
        plt.yticks(fontsize=tick_fontsize)
        plt.tight_layout()

        self._savePlot(
            save_plot=save_plot,
            save_path=save_path,
            save_fname=save_fname,
            dpi=dpi,
            save_message="Saved feature importance plot"
        )

        plt.show()

    def plotMultiTaskPerformance(
            self,
            data: pd.DataFrame,
            x_col: str,
            y_col: str,
            ascending: bool=False,
            top_n: int=None,
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
        
        """
        Plots cross-feature performance, showing how models trained on one feature representation 
        perform when predicting features from another representation.

        Parameters
        ----------
        data: pd.DataFrame
                            Dataframe containing the cross-feature performance
        x_col: str
                            Column containing the performance metric to be plotted
        y_col: str
                            Column containing the features predicted on
        ascending: bool (optional)
                            Order to sort the data. True = low to high, False = high to low
        top_n: int (optional)
                            Changes the number of features plotted
        subset: str (optional)                            
                            Set this to look at the descriptors which contain this string
        figsize: tuple (optional)
                            Changes the size of the figure
        title: str (optional)
                            Sets the title of the plot
        title_fontsize: int (optional)
                            Changes the font size for the title
        label_fontsize: int (optional)
                            Changes the font size for the axis labels
        tick_fontsize: int (optional)
                            Changes the font size for the axis ticks                            
        save_plot: bool (optional)
                            Flag to save the plot
        save_path: str, Path (optional)
                            Path to save the plot to
        save_fname: str (optional)
                            Name to save the plot under. Must end in image extension (.png)
        dpi: int (optional)
                            Dots per inch, or quality, of saved image
        """
        
        # Look for subset of desciptors containing the string "subset"
        if subset:
            data = data[data.index.astype(str).str.contains(subset, case=False, na=False)]

        # Ensure numerical values, drop N/A and sort values
        data = data.copy()
        data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
        data = data.dropna(subset=[x_col])
        data = data.sort_values(by=x_col, ascending=ascending)

        # If specified, look at top_n number of features
        if top_n:
            data = data.head(top_n)


        # Plotting the figure
        plt.figure(figsize=figsize)
        plt.barh(data.index, data[x_col].values)
        plt.title(title, fontsize=title_fontsize)

        plt.xlabel(x_col.capitalize(), fontsize=label_fontsize)
        plt.ylabel(y_col, fontsize=label_fontsize)

        plt.yticks(fontsize=tick_fontsize)
        plt.xticks(fontsize=tick_fontsize)

        plt.tight_layout()

        self._savePlot(
            save_plot=save_plot,
            save_path=save_path,
            save_fname=save_fname,
            dpi=dpi,
            save_message="Saved multi-task plot"
        )

        plt.show()

    def computeGroupPerf(
            self,
            data: pd.DataFrame,
            descriptor_groups: dict,
            metrics: list[str],
            exclude: list[str]=None
        ) -> pd.DataFrame:

        """
        Compute average performance per descriptor group.
        
        Parameters
        ----------
        data: pd.DataFrame
                            Data containing the descriptors and performance metrics
        descriptor_groups: dict
                            Dictionary of descriptors grouped by user (can use getGroups function
                            for rdkit and mordred)
        metrics: list[str]
                            Metric/s to average across descriptor groups
        exclude: list[str] (optional)
                            Descriptors to exclude in the averaging calculation
        
        Returns
        -------
        pd.DataFrame:
                    Contains average metrics across each descriptor group
        
        """
        
        # Exclude descriptors, if specified
        if exclude is None:
            exclude = []

        # Calculate group performances
        group_perf = {}
        for group, descs in descriptor_groups.items():
            valid = [d for d in descs if d in data.index and d not in exclude]
            if valid:
                group_perf[group] = data.loc[valid, metrics].mean()
        return pd.DataFrame.from_dict(group_perf, orient="index", columns=[f"avg_{metrics}"])

    def plotGroupRadar(
            self, 
            group_perf_df: pd.DataFrame,
            title: str="Radar Plot",
            figsize: tuple=(10,10),
            title_fontsize: int=16,
            label_fontsize: int=12,
            tick_fontsize: int=10,
            save_plot: bool=False,
            save_path: Union[str, Path]=None,
            save_fname: str="group_radar.png",
            dpi: int=400
    
        ):
        """
        Make a radar plot of group performances with dual scales (0–1 and 0.5–1).
        
        Parameters
        ----------        
        group_perf_df: pd.DataFrame
                    Dataframe containing the descriptor group performances, calculated
                    by computeGroupPerf function
        title: str (optional)
                    Sets the title of the plot
        figsize: tuple (optional)
                            Changes the size of the figure
        title_fontsize: int (optional)
                            Changes the font size for the title
        label_fontsize: int (optional)
                    Changes the font size for the axis labels
        tick_fontsize: int (optional)
                            Changes the font size for the axis ticks    
        save_plot: bool (optional)
                            Flag to save the plot
        save_path: str, Path (optional)
                            Path to save the plot to
        save_fname: str (optional)
                            Name to save the plot under
        dpi: int (optional)
                            Dots per inch, or quality, of saved image
        """
        

        labels = group_perf_df.index.tolist()
        values = group_perf_df.iloc[:,0].values
        values = np.append(values, values[0])
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        angles = np.append(angles, angles[0])

        # Set colour palette
        palette = sns.color_palette("tab10")
        blue, red = palette[0], palette[3]

        # Create the figure
        fig = plt.figure(figsize=figsize)

        # Axis 1 (0–1)
        ax1 = fig.add_subplot(111, polar=True)
        ax1.plot(
            angles, 
            values, 
            linewidth=2, 
            color=blue, 
            label="0–1 scale",
            alpha=0.5, 
            zorder=1
            )
        ax1.fill(angles, values, alpha=0.2, color=blue, zorder=0)

        # Configuring the y-ticks
        ax1.set_ylim(0,1)
        ticks_blue = np.arange(0,1.1,0.1)
        ax1.set_yticks(ticks_blue)
        ax1.set_yticklabels([f"{t:.1f}" if t>0 else "" for t in ticks_blue], 
                            color=blue, fontsize=tick_fontsize, fontweight="bold")
        
        # Configuring the x-ticks
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(labels, fontsize=label_fontsize, fontweight="bold")
        ax1.set_rlabel_position(90)

        # Axis 2 (0.5–1)
        ax2 = fig.add_subplot(111, polar=True, frame_on=False)
        ax2.plot(
            angles, 
            values, 
            linewidth=2, 
            color=red, 
            linestyle="--", 
            label="0.5–1 scale", 
            alpha=0.5, zorder=1
            )
            
        ax2.fill(angles, values, alpha=0.15, color=red, zorder=0)

        # Configuring the y-ticks
        ax2.set_ylim(0.5,1)
        ticks_red = np.arange(0.5,1.01,0.1)
        ax2.set_yticks(ticks_red)
        ax2.set_yticklabels([f"{t:.2f}" for t in ticks_red], 
                            color=red, fontsize=title_fontsize, fontweight="bold")
        
        # Configuring the x-ticks
        ax2.set_xticks([])
        ax2.set_xticklabels([])
        ax2.set_rlabel_position(270)

        # Merge legends
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(
            handles1+handles2, 
            labels1+labels2, 
            loc="upper right", 
            bbox_to_anchor=(1.2,1.1), 
            fontsize=label_fontsize
            )

        plt.title(title, fontsize=title_fontsize, weight="bold", pad=40)

        self._savePlot(
            save_plot=save_plot,
            save_path=save_path,
            save_fname=save_fname,
            dpi=dpi,
            save_message="Saved descriptor-grouped radar plot"
        )

    def plotGroupBar(
            self, 
            *group_dfs: pd.DataFrame,
            labels: list[str],
            x_label: str = "Descriptor Group",
            y_label: str = "Average Pearson r",
            title: str = "Group Comparison",
            figsize: tuple = (12,6),
            rotation: int = 45,
            title_fontsize: int=16,
            label_fontsize: int=12,
            tick_fontsize: int=10,
            save_plot: bool=False,
            save_path: Union[str, Path]=None,
            save_fname: str="grouped_barchart.png",
            dpi: int=400
        ):

        """
        Grouped barplot for any number of models' performance.

        Parameters
        ----------

        *group_dfs: pd.DataFrame
                        Dataframes containing the descriptor group performances, calculated
                        by computeGroupPerf function
        labels: list[str]
                        Labels for the dataframes (in order of definition)
        x_label: str
                        X-axis label
        y_label: str
                        Y-axis label
        title: str (optional)
                        Sets the title of the plot
        figsize: tuple (optional)
                        Changes the size of the figure
        rotation: int (optional)
                        Degrees for labels to be rotated by (45 = 45 degrees)
        title_fontsize: int (optional)
                            Changes the font size for the title
        label_fontsize: int (optional)
                            Changes the font size for the axis labels
        tick_fontsize: int (optional)
                            Changes the font size for the axis ticks
        save_plot: bool (optional)
                            Flag to save the plot
        save_path: str, Path (optional)
                            Path to save the path to
        save_fname: str (optional)
                            Name to save the plot under. Must end in image extension (.png)    
        dpi: int (optional)
                            Dots per inch, or quality, of saved image                                

        """

        # Ensure labels are created or match number of dataframes provided
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

        # Get colour from preset colour scheme
        palette = [self._getColour(l) for l in labels]

        # Initialise the figure
        plt.figure(figsize=figsize)
        sns.barplot(
            data=all_df,
            x="group",
            y=all_df.columns[0],
            hue="dataset",
            palette=palette
        )
        plt.xticks(rotation=rotation, ha="right", fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.ylabel(y_label, fontsize=label_fontsize, weight="bold")
        plt.xlabel(x_label, fontsize=label_fontsize, weight="bold")
        plt.title(title, fontsize=title_fontsize, weight="bold")
        plt.legend(fontsize=label_fontsize)
        plt.tight_layout()

        self._savePlot(
            save_plot=save_plot,
            save_path=save_path,
            save_fname=save_fname,
            dpi=dpi,
            save_message="Saved descriptor-grouped bar plot"
        )
        plt.show()

    def plotBar(
            self,
            data: pd.DataFrame,
            x_label: str,
            y_label: str,
            title: str=" ",
            figsize: tuple=(12,6),
            rotation: int=45,
            title_fontsize: int=16,
            label_fontsize: int=12,
            tick_fontsize: int=10,
            save_plot: bool=False,
            save_path: Union[str, Path]=None,
            save_fname: str="grouped_barchart.png",
            dpi: int=400
    ):
        
        """
        Generic function to plot a bar chart

        Parameters
        ----------
        
        data: pd.DataFrame
                        Dataframe containing data to be plotted
        x_label: str 
                        Column containing the names of bars to be plotted
        y_label: str
                        Column containing numerical values
        title: str (optional)
                        Sets title for the bar chart
        figsize: tuple (optional)
                            Changes the size of the figure        
        rotation: int (optional)
                        Degrees for labels to be rotated by (45 = 45 degrees)
        title_fontsize: int (optional)
                            Changes the font size for the title
        label_fontsize: int (optional)
                            Changes the font size for the axis labels
        tick_fontsize: int (optional)
                            Changes the font size for the axis ticks      
        save_plot: bool (optional)
                            Flag to save the plot
        save_path: str, Path (optional)
                            Path to save the path to
        save_fname: str (optional)
                            Name to save the plot under. Must end in image extension (.png)
        dpi: int (optional)
                            Dots per inch, or quality, of saved image
        """
        
        # Initialise the figure
        plt.figure(figsize=figsize)
        sns.barplot(
            data=data,
            x=x_label,
            y=data[y_label],
            palette="tab10"
        )
        plt.xticks(rotation=rotation, ha="right", fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.ylabel(y_label, fontsize=label_fontsize, weight="bold")
        plt.xlabel(x_label, fontsize=label_fontsize, weight="bold")
        plt.title(title, fontsize=title_fontsize, weight="bold")
        plt.legend(fontsize=label_fontsize)
        plt.tight_layout()

        self._savePlot(
            save_plot=save_plot,
            save_path=save_path,
            save_fname=save_fname,
            dpi=dpi,
            save_message="Saved bar chart"
        )

        plt.show()

    def plotBoxPlots(
            self,
            *data_dfs: pd.DataFrame,
            trained_labels: list[str],
            predicted_labels: list[str],
            title: str= " ",
            x_label: str="Models",
            y_label: str="Pearson_r",
            figsize=(16, 8),
            rotation: int=45,
            title_fontsize: int=16,
            label_fontsize: int=12,
            tick_fontsize: int=10,            
            save_plot: bool=False,
            save_path: Path=Path("./"),
            save_fname: str="box_plot.png",
            dpi: int=400
    ):
        
        """
        Function to generate box plots specifically for the embedding/descriptor data 
        generated in this project
        
        Parameters
        ----------
        *data_dfs: pd.DataFrame
                            Dataframes containing the numerical data to generate boxplots
        trained_labels: list[str]
                            Names of the feature sets models were trained on
        predicted_labels: list[str]
                            Names of feature sets models were predicting
        title: str (optional)
                            Sets the title of the plot
        x_label: str
                            Sets the X-axis label                               
        y_label: str
                            Column containing numerical values
        figsize: tuple (optional)
                            Changes the size of the figure     
        rotation: int (optional)
                            Degrees for labels to be rotated by (45 = 45 degrees)
        title_fontsize: int (optional)
                            Changes the font size for the title
        label_fontsize: int (optional)
                            Changes the font size for the axis labels
        tick_fontsize: int (optional)
                            Changes the font size for the axis ticks         
        save_plot: bool (optional)
                            Flag to save the plot
        save_path: str, Path (optional)
                            Path to save the path to
        save_fname: str (optional)
                            Name to save the plot under. Must end in image extension (.png)
        dpi: int (optional)
                            Dots per inch, or quality, of saved image
        """

        # Ensure number of dataframes and labels input match
        if not (len(data_dfs) == len(trained_labels) == len(predicted_labels)):
            raise ValueError("data_dfs, trained_labels, and predicted_labels must all have the same length")

        
        # Combine data
        combined = []
        for df, tr_label, pred_label in zip(data_dfs, trained_labels, predicted_labels):
            temp = df.copy()
            temp["trained_on"] = tr_label
            temp["predicted_on"] = pred_label
            dataset = f"{tr_label}-{pred_label}"
            temp["dataset"] = dataset
            
            # Adding suffix to ensure all feature_names are unique and identifiable
            temp.index = temp.index.astype(str) + f"_{dataset}"

            combined.append(temp)

        combined_df = pd.concat(combined, axis=0)

        # Get colour from preset colour scheme
        tr_palette = [self._getColour(l) for l in trained_labels]
        pred_palette = [self._getColour(l) for l in predicted_labels]

        # Initialise figure
        fig, ax = plt.subplots(figsize=figsize)

        # Draw background rectangles (predicted colors)
        y_min, y_max = combined_df[y_label].min(), combined_df[y_label].max()
        y_range = y_max - y_min
        datasets = combined_df["dataset"].unique()

        # Place rectangles on figure
        for i, (dataset, color) in enumerate(zip(datasets, pred_palette)):
            rect = patches.Rectangle(
                (i - 0.5, y_min - 0.05 * y_range),
                width=0.995,
                height=y_range * 1.1,
                color=color,
                alpha=0.25,
                zorder=0
            )
            ax.add_patch(rect)

        # Plot the boxplots (training colors)
        sns.boxplot(
            data=combined_df,
            x="dataset",
            y=y_label,
            palette=tr_palette[:len(datasets)],
            ax=ax,
            width=0.6,
            zorder=1,
            hue="dataset",
            legend=False
        )

        # Legend setup
        # Global legend (background vs box color)
        legend_patches = [
            patches.Patch(facecolor="grey", alpha=0.25, label="Predicted on (background)"),
            patches.Patch(facecolor="grey", label="Trained on (box color)")
        ]

        # Color key for training and predicted sources
        unique_sources = list(set(trained_labels + predicted_labels))
        color_key_patches = [
            patches.Patch(color=self._getColour(src), label=src) for src in unique_sources
        ]

        # Combine the legends
        first_legend = plt.legend(
            handles=legend_patches,
            loc="upper left",
            bbox_to_anchor=(1.02, 0.75),
            frameon=False,
            title="Colour meaning",
            fontsize=label_fontsize,
            title_fontsize=label_fontsize + 1
        )
        ax.add_artist(first_legend)

        plt.legend(
            handles=color_key_patches,
            loc="lower left",
            bbox_to_anchor=(1.02, 0.25),
            frameon=False,
            title="Colour key",
            fontsize=label_fontsize,
            title_fontsize= + 1
        )

        # Labels & formatting
        ax.set_title(title, fontsize=title_fontsize + 2, weight="bold", pad=15)
        ax.set_xlabel(x_label, fontsize=label_fontsize)
        ax.set_ylabel(y_label, fontsize=label_fontsize)
        plt.xticks(rotation=rotation, ha="right", fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.tight_layout(rect=[0, 0, 0.7, 1])
        plt.show()

        self._savePlot(
            save_plot=save_plot,
            save_path=save_path,
            save_fname=save_fname,
            dpi=dpi,
            save_message="Saved descriptor-grouped bar plot"
        )

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
            ascending: bool=False,
            title_fontsize: int=16,
            label_fontsize: int=12,
            tick_fontsize: int=10,            
            save_plot: bool=False,
            save_path: Path=Path("./"),
            save_fname: str="box_plot.png",
            dpi: int=400
        ):
        """
        Plot one heatmap per descriptor group.
        
        Descriptors
        -----------
        data: pd.DataFrame

        descriptor_groups: dict

        metric: str

        title_prefix: str (optional)
                            Sets the title predix of the plot. The rest of the title will 
                            contain the group name for the descriptors

        cmap: str (optional)
                            Colour of the palette to use
        annot: bool (optional)

        vmin: float (optional)

        vmax: float (optional)

        ascending: bool (optional)
                            Order to sort the data. True = low to high, False = high to low
        title_fontsize: int (optional)
                            Changes the font size for the title
        label_fontsize: int (optional)
                            Changes the font size for the axis labels
        tick_fontsize: int (optional)
                            Changes the font size for the axis ticks                            
        save_plot: bool (optional)
                            Flag to save the plot
        save_path: str, Path (optional)
                            Path to save the path to
        save_fname: str (optional)
                            Name to save the plot under. Must end in image extension (.png)
        dpi: int (optional)
                            Dots per inch, or quality, of saved image

        """

        # Iterate through the groups and descriptors
        for group, descs in descriptor_groups.items():
            valid = [d for d in descs if d in data.index]
            if not valid:
                continue
            
            # Sort metrics
            df_group = data.loc[valid, [metric]].sort_values(metric, ascending=ascending)

        # Initialise figure
            plt.figure(figsize=(6, max(4, len(valid) * 0.25)))
            sns.heatmap(df_group,
                        cmap=cmap,
                        annot=annot,
                        fmt=".2f",
                        cbar_kws={"label": metric},
                        vmin=vmin,
                        vmax=vmax,
                        annot_kws={"fontsize": tick_fontsize})
            plt.title(f"{title_prefix}: {group}", fontsize=title_fontsize, weight="bold", pad=20)
            plt.ylabel("Descriptor", fontsize=label_fontsize, weight="bold")
            plt.xlabel("")
            plt.yticks(fontsize=tick_fontsize)
            plt.tight_layout()


            self._savePlot(
                save_plot=save_plot,
                save_path=save_path,
                save_fname=f"{save_fname}_{group}.png",
                dpi=dpi,
                save_message=f"Saved {group} heatmap"
            )
            
            plt.show()

    def plotModelPerformanceBars(
            self,
            base_path: Path,
            model_jsons: dict[str, Path],
            model_labels: list[str],
            metrics: list[str] = ["r2", "Pearson_r", "RMSE", "Bias"],
            figsize: tuple = (6, 4),
            title_fontsize: int = 16,
            label_fontsize: int = 12,
            tick_fontsize: int = 10,
            save_plot: bool = False,
            save_path: Path = Path("./"),
            save_fname: str = "model_performance.png",
            dpi: int = 400,
    ):
        """
        Function to generate individual bar plots comparing model performance metrics 
        (e.g., R², Pearson r, RMSE, Bias) across multiple trained models.

        Parameters
        ----------
        base_path : Path
                        Base directory containing all model result folders.
        model_jsons : dict[str, Path]
                        Dictionary mapping model names to their respective JSON performance files.
        model_labels : list[str]
                        Names of the models (used for coloring and ordering in the plot).
        metrics : list[str] (optional)
                        List of performance metrics to plot. Each metric will be plotted in a separate figure.
        figsize : tuple, (optional)
                        Figure size for each individual plot.
        title_fontsize : int, (optional)
                        Font size for the plot titles.
        label_fontsize : int, (optional)
                        Font size for the axis labels.
        tick_fontsize : int, (optional)
                        Font size for the tick labels.
        save_plot : bool, (optional)
                        Flag to save the plots as image files.
        save_path : Path, (optional)
                        Directory path to save the generated plots.
        save_fname : str, (optional)
                        Base filename to save the plots under. Must end with an image extension (e.g., .png).
        dpi : int, (optional)
                        Dots per inch (image resolution) for the saved plots.

        """

        # --- Load JSONs into DataFrame ---
        records = []
        for model, file in model_jsons.items():
            file_path = base_path / file if not file.is_absolute() else file
            with open(file_path, "r") as f:
                data = json.load(f)
                data["Model"] = model
                records.append(data)

        perf_df = pd.DataFrame(records).set_index("Model")

        # --- Setup colors ---
        colours = [self._getColour(l) for l in model_labels]

        # --- Plot metrics ---
        for metric in metrics:
            plt.figure(figsize=figsize)
            bars = plt.bar(
                perf_df.index,
                perf_df[metric],
                color=colours,
                edgecolor="black",
            )
            plt.title(f"{metric} Comparison", fontsize=title_fontsize, weight="bold")
            plt.ylabel(metric, fontsize=label_fontsize)
            plt.xlabel("Model", fontsize=label_fontsize)
            plt.xticks(rotation=45, ha="right", fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            plt.tight_layout()

            self._savePlot(
                save_plot=save_plot,
                save_path=save_path,
                save_fname=f"{save_fname}_{metric}.png",
                dpi=dpi,
                save_message=f"Saved model performance bar charts"
            )
        
            plt.show()

    def plotPCA(
        self,
        train,
        prediction,
        source_ls: list[str],
        validation: str | pd.DataFrame = None,
        n_components: int = 5,
        loadings_filename: str = "pca_loadings",
        pca_df_filename: str = "pca_components",
        contamination: float = 0.00001,
        plot_fname: str = "pca_plot",
        save_plot: bool = False,
        save_dir: Path=Path("./"),
        save_extra_data: bool = False,
        plot_area: bool = False,
        plot_scatter: bool = True,
        random_seed: int = None,
        plot_loadings: bool = False,
        plot_title: str = None,
        remove_outliers: bool = True,
        kdep_sample_ls: list[str] = None,
        axis_fontsize: int = 18,
        label_fontsize: int = 16,
        legend_fontsize: int = 14,
    ):
        if random_seed is None:
            random_seed = rand.randint(0, 2**31)

        kdep_sample_ls = kdep_sample_ls or []

        # Load train
        train_df = pd.read_csv(train, index_col="ID") if isinstance(train, (str, Path)) else train.copy()
        train_df["Source"] = source_ls[0]

        # Load prediction(s)
        if isinstance(prediction, str) and "*" in prediction:
            prediction_df = pd.concat([pd.read_csv(f, index_col="ID") for f in glob(prediction)], axis=0)
        elif isinstance(prediction, (str, Path)):
            prediction_df = pd.read_csv(prediction, index_col="ID")
        else:
            prediction_df = prediction.copy()
        prediction_df["Source"] = source_ls[-1]

        # Validation (optional)
        if validation is not None:
            validation_df = (
                pd.read_csv(validation, index_col="ID")
                if isinstance(validation, (str, Path))
                else validation.copy()
            )
            validation_df["Source"] = source_ls[1]
        else:
            validation_df = pd.DataFrame()

        # --- Common columns
        common_cols = set(train_df.columns) & set(prediction_df.columns)
        if not validation_df.empty:
            common_cols &= set(validation_df.columns)
        common_cols.discard("Source")

        common_cols = list(common_cols)
        train_df = train_df[common_cols + ["Source"]]
        prediction_df = prediction_df[common_cols + ["Source"]]
        if not validation_df.empty:
            validation_df = validation_df[common_cols + ["Source"]]

        # Combine all datasets
        dfs = [train_df, validation_df, prediction_df]
        dfs = [df for df in dfs if not df.empty]
        combined_df = pd.concat(dfs, axis=0).dropna()

        # --- Scale & PCA
        scaler = StandardScaler()
        scaled = pd.DataFrame(
            scaler.fit_transform(combined_df[common_cols]),
            columns=common_cols,
            index=combined_df.index,
        )
        scaled["Source"] = combined_df["Source"]

        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled[common_cols])
        explained_var = pca.explained_variance_ratio_ * 100

        # --- Loadings
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings_df = pd.DataFrame(loadings, columns=[f"PC{i+1}" for i in range(n_components)], index=common_cols)
        abs_loadings_df = loadings_df.abs().rename_axis("Features")

        if save_extra_data:
            loadings_df.to_csv(save_dir / f"{loadings_filename}.csv", index_label="Features")
            abs_loadings_df.to_csv(save_dir / f"{loadings_filename}_abs.csv", index_label="Features")

        # --- PCA dataframe
        pca_df = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(n_components)], index=combined_df.index)
        pca_df["Source"] = combined_df["Source"]

        if save_extra_data:
            pca_df.to_csv(save_dir / f"{pca_df_filename}.csv.gz", index_label="ID", compression="gzip")

        # --- Outlier removal
        if remove_outliers:
            lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
            mask = lof.fit_predict(pca_df[[f"PC{i+1}" for i in range(n_components)]])
            pca_df = pca_df[mask == 1]

        # --- Colors
        palette = sns.color_palette('dark')
        source_colors = {src: palette[i % len(palette)] for i, src in enumerate(source_ls)}

        # --- Subplots
        fig, axs = plt.subplots(n_components, n_components, figsize=(10, 10))

        for i in range(n_components):
            for j in range(n_components):
                if i == j:
                    # KDE diagonal
                    sns.kdeplot(
                        x=f"PC{i+1}",
                        hue="Source",
                        data=pca_df,
                        common_norm=False,
                        fill=True,
                        ax=axs[i, i],
                        palette=source_colors,
                        legend=False,
                    )
                    axs[i, i].set_ylabel("Density", fontsize=label_fontsize)
                else:
                    if plot_scatter:
                        sns.scatterplot(
                            x=f"PC{j+1}",
                            y=f"PC{i+1}",
                            hue="Source",
                            data=pca_df,
                            ax=axs[i, j],
                            palette=source_colors,
                            legend=False,
                            alpha=0.2,
                            s=10
                        )
                    if plot_area:
                        for src in pca_df["Source"].unique():
                            pts = pca_df[pca_df["Source"] == src][[f"PC{j+1}", f"PC{i+1}"]].values
                            if len(pts) > 3:
                                hull = ConvexHull(pts)
                                hull_pts = np.append(hull.vertices, hull.vertices[0])
                                axs[i, j].fill(
                                    pts[hull_pts, 0], pts[hull_pts, 1],
                                    alpha=0.2, color=source_colors[src], edgecolor=source_colors[src]
                                )

                # Labels
                if i == n_components - 1:
                    axs[i, j].set_xlabel(f"PC{j+1} ({explained_var[j]:.1f}% var)", fontsize=axis_fontsize)
                if j == 0:
                    axs[i, j].set_ylabel(f"PC{i+1} ({explained_var[i]:.1f}% var)", fontsize=axis_fontsize)

        # --- Legend
        handles = [
            plt.Line2D([], [], color=source_colors[src], marker="o", linestyle="None", markersize=8, label=src)
            for src in pca_df["Source"].unique()
        ]
        fig.legend(handles=handles, loc="upper center", ncol=len(source_ls), fontsize=legend_fontsize, frameon=False)
        fig.suptitle(plot_title, fontsize=axis_fontsize + 4, y=1.02)
        plt.tight_layout()

        if save_plot:
            fig.savefig(save_dir / f"{plot_fname}.png", dpi=600, bbox_inches="tight")

        # --- Optional loadings plot
        if plot_loadings:
            loadings_df["Max"] = loadings_df.abs().max(axis=1)
            sig_loadings = loadings_df[loadings_df["Max"] > 0.3].drop(columns="Max")
            fig2, axes = plt.subplots(n_components, 1, figsize=(20, 18), sharex=True)
            for n in range(1, n_components + 1):
                sns.barplot(x=sig_loadings.index, y=sig_loadings[f"PC{n}"], ax=axes[n - 1])
                axes[n - 1].set_ylabel(f"PC{n} Loadings", fontsize=label_fontsize)
            plt.xticks(rotation=90)
            fig2.tight_layout()
            fig2.savefig(save_dir / f"{plot_fname}_loadings.png", dpi=600, bbox_inches="tight")

        return fig, pca_df, loadings_df
    
    def plotPredictions(
            self,
    ):
        return