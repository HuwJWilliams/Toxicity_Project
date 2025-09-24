# %%
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# %% ========= Constants =========

FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[2]
DATASET_DIR = PROJ_DIR / "datasets" / "LD50"
DATA_PATH = DATASET_DIR / "ld50_molecules.csv"

# %% ========= Loading data =========

raw_data = pd.read_csv(DATA_PATH)

# %% ========= Plot raw values =========
# Count how many rows per organism
counts = raw_data['Organism'].value_counts().sort_index()

plt.figure(figsize=(10,6))
counts.plot(kind="bar", color="skyblue", edgecolor="black")

plt.xlabel("Organism")
plt.ylabel("Count")
plt.title("Number of LD50 records per organism")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %% ========= Plot values for animals defined=========
animals = ["mouse","rat"]
data = raw_data[raw_data['Organism'].isin(animals)]

# Count number of records per administration route
counts = data["Administration route"].value_counts()

plt.figure(figsize=(10,6))
counts.plot(kind="bar", color="skyblue", edgecolor="black")

plt.xlabel("Administration route")
plt.ylabel("Count")
plt.title("Number of LD50 records by administration route (mouse & rat)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %% ========= Plot raw oral values =========
oral = data[data["Administration route"] == "oral"]
oral_raw = oral['LD50 {measured, converted}'].dropna().tolist()

# Compute stats
min_val = np.min(oral_raw)
max_val = np.max(oral_raw)
avg_val = np.mean(oral_raw)

p90, p95, p99 = np.percentile(oral_raw, [90, 95, 99])

plt.figure(figsize=(6,4))
plt.hist(oral_raw, bins=50, color="lightcoral", edgecolor="black", alpha=0.7)
plt.xlabel("LD50 (oral, raw)")
plt.ylabel("Frequency")
plt.title("Raw Oral LD50 Distribution")
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Add stats box in top-right
textstr = (
    f"Min: {min_val:.2f}\n"
    f"Max: {max_val:.2f}\n"
    f"Avg: {avg_val:.2f}\n"
    f"P90: {p90:.2f}\n"
    f"P95: {p95:.2f}\n"
    f"P99: {p99:.2f}"
)
plt.gca().text(
    0.95, 0.95, textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.tight_layout()
plt.show()

# %% ========= 2) Filter out values > 9500 =========
oral_filtered = oral[oral['LD50 {measured, converted}'] <= 9500].copy()
len(oral_filtered)

plt.figure(figsize=(6,4))
plt.hist(
    oral_filtered['LD50 {measured, converted}'], 
    bins=50, color="skyblue", edgecolor="black", alpha=0.7
)
plt.xlabel("LD50 (oral, ≤ 9,500 mg/kg)")
plt.ylabel("Frequency")
plt.title("Filtered Oral LD50 Distribution")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
# %%

# %% ========= Save filtered dataframe =========
out_path = DATASET_DIR / "ld50_rat-mouse_oral_lt_9500.csv"
oral_filtered = oral_filtered.reset_index(drop=True)
oral_filtered.index = [f"tox-{i+1}" for i in range(len(oral_filtered))]
oral_filtered.to_csv(out_path, index=True)
print(f"Filtered data saved -> {out_path}")
# %%
len(oral_filtered)
len(oral_raw)

# %%
len(oral_filtered)/len(raw_data)

# %%
