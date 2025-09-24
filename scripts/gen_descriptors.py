from models.transfer_model import TL
import pandas as pd
from pathlib import Path

FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[1]

DATA_PATH = Path(PROJ_DIR / "datasets"/"LD50"/"ld50_rat-mouse_oral_lt_9500.csv")
DESC_PATH = Path(PROJ_DIR / "datasets"/"descriptors"/"rdkit_descriptors.csv")

df = pd.read_csv(DATA_PATH, index_col="ID")

model = TL(tokeniser=None, encoder=None, log_identifier="desc_calc")

desc_df = model.calculateDescriptors(smiles=df["SMILES"], descriptor_set="rdkit")
desc_df.to_csv(DESC_PATH, index_label="ID")
