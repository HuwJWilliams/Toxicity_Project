#%%
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch
import numpy as np
from rdkit import Chem
import pandas as pd
import random
import sys
from rdkit.Chem import Descriptors
from sklearn.model_selection import KFold, GridSearchCV
from mordred import Calculator, descriptors
import logging
import joblib
import json

FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[2]
DATASET_DIR = PROJ_DIR / "datasets"
PALMERCHEM_SOFTWARE = Path.home() / "PalmerChem_Software" / "src" / "models"
LOG_DIR = PROJ_DIR / "scripts" / "models" / "logs"
print(PALMERCHEM_SOFTWARE)

sys.path.insert(0, str(PALMERCHEM_SOFTWARE))
from RFRegressor import RFRegressor

# ========== Constants ========== #
tok_ls          = ["ibm/MoLFormer-XL-both-10pct", "DeepChem/ChemBERTa-100M-MLM"]
TOKENIZER       = tok_ls[0]
mod_ls          = [ "ibm/MoLFormer-XL-both-10pct", "DeepChem/ChemBERTa-100M-MLM"]
MODEL           = mod_ls[0]
DATA            = DATASET_DIR / "LD50" / "ld50_rat-mouse_oral_lt_9500.csv"
MAX_LEN         = 400         # shorter = cheaper RAM/CPU
BATCH_SIZE      = 64
SEED            = 42
LOG_LEVEL       = logging.DEBUG

# ============ Class ============ #
class TL():
    def __init__(
            self,
            tokeniser,
            encoder,
            unembedded_df: pd.DataFrame=None,
            embedded_df: pd.DataFrame=None,
            seed: int=None,
            log_name: str = "TLModel",
            log_to_file: bool=True,
            log_dir: Path = LOG_DIR,
            log_level= LOG_LEVEL,
            log_identifier: str = ""
            ):
        
        self.tokeniser = tokeniser
        self.encoder = encoder
        self.unembedded_df = unembedded_df
        self.embedded_df = embedded_df
        self.train_df = None
        self.test_df = None
        self.seed=seed

        #===== Logger =====#
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s | %(funcName)s | %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # File handler
            if log_to_file:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / f"{log_name}_{log_identifier}.log"
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

        self.logger.info("Logger initialised.")

    
    def rng(self) -> int:
        if self.seed is None:
            seed = random.randint(0, 2**32)
        else:
            seed = self.seed
        
        self.logger.info(f"Random seed: {seed}")
        return seed

    def embedSMILESChemBERTa(
            self, 
            smiles_col: str, 
            max_len: int, 
            batch_size: int,
            save_path: Path=None
    ) -> pd.DataFrame:
        
        self.logger.info("Embedding SMILES using ChemBERTa")
        self.logger.debug(f"Tokeniser:\n{TOKENIZER}\nEncoder:\n{MODEL}")
        
        smiles_list = self.unembedded_df[smiles_col].tolist()
        
        # Initialise empty list for embeddings to reside
        embeddings = []

        # Loop over smiles in batches
        self.logger.info("Looping over SMILES in batches")
        total_batches = len(smiles_list) // batch_size + (len(smiles_list) % batch_size > 0)

        for i in range(0, len(smiles_list), batch_size):
            batch = [str(s) for s in smiles_list[i:i+batch_size]]
            current_batch_no = i // batch_size + 1

            # Create tokens from SMILES
            enc = self.tokeniser(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
                add_special_tokens=True)
            
            # Forward pass through encoder
            with torch.no_grad():
                hidden = self.encoder(**enc).last_hidden_state
                mask   = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            embeddings.append(pooled.cpu().numpy().astype(np.float32))

            self.logger.info(f"Processed batch {current_batch_no} of {total_batches}")


        if embeddings:
            arr = np.vstack(embeddings)
        else:
            arr = np.zeros((0, self.encoder.config.hidden_size), np.float32)

        emb_df = pd.DataFrame(
            arr,
            index=self.unembedded_df.index,
            columns=[f"emb_{i}" for i in range(1, arr.shape[1] + 1)]
        )

        self.logger.info(f"Embeddings data frame created with a shape: {emb_df.shape}")
        
        if save_path:
            if save_path.exists():
                self.logger.warning(f"File already exists: {save_path}. Skipping save.")
            else:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                emb_df.to_csv(save_path, index_label='ID')
                self.logger.info(f"Embeddings saved to: {save_path}")
            
        return emb_df
    
    def embedSMILESMolFormer(
        self,
        smiles_col: str,
        batch_size: int,
        save_path: Path=None
    ) -> pd.DataFrame:

        self.logger.info("Embedding SMILES using MolFormer")
        self.logger.debug(f"Tokeniser:\n{self.tokeniser}\nEncoder:\n{self.encoder}")

        smiles_list = self.unembedded_df[smiles_col].tolist()
        embeddings = []

        self.encoder.eval()
        device = next(self.encoder.parameters()).device

        # Loop over smiles in batches
        self.logger.info("Looping over SMILES in batches")
        total_batches = len(smiles_list) // batch_size + (len(smiles_list) % batch_size > 0)

        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            current_batch_no = i // batch_size + 1

            # === Preprocessing step: moved internally ===
            inputs = self.tokeniser(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self.encoder(**inputs)

                if isinstance(output, torch.Tensor):
                    pooled = output
                elif hasattr(output, "last_hidden_state"):
                    pooled = output.last_hidden_state[:, 0, :]
                elif hasattr(output, "pooler_output"):
                    pooled = output.pooler_output
                else:
                    raise RuntimeError("Unknown output structure from encoder")

            embeddings.append(pooled.cpu().numpy().astype(np.float32))
            self.logger.info(f"Processed batch {current_batch_no} of {total_batches}")

        arr = np.vstack(embeddings)
        emb_df = pd.DataFrame(
            arr,
            index=self.unembedded_df.index,
            columns=[f"emb_{i}" for i in range(1, arr.shape[1] + 1)]
        )

        if save_path:
            if save_path.exists():
                self.logger.warning(f"File already exists: {save_path}. Skipping save.")
            else:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                emb_df.to_csv(save_path, index_label='ID')
                self.logger.info(f"Embeddings saved to: {save_path}")

        self.logger.info(f"Embeddings data frame created with shape: {emb_df.shape}")
        return emb_df

    def calculateDescriptors(
            self, 
            smiles: pd.Series,
            descriptor_set: str="rdkit",
            ignore_3D = True,
            drop_constant: bool=True,
            drop_fragments: bool=True,
            drop_classes: bool=True,
        ) -> pd.DataFrame:

        descriptor_set = descriptor_set.lower()
        self.logger.info(f"Creating {descriptor_set} descriptors")

        parsed = []

        self.logger.debug(f"Total length of smiles input: {len(smiles)}")
        for mol_id, smi in smiles.items():
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                parsed.append((mol_id, smi, mol))
        if not parsed:
            return pd.DataFrame(columns=["SMILES"])

        ids = [p[0] for p in parsed]
        smis = [p[1] for p in parsed]
        mols = [p[2] for p in parsed]
        
        self.logger.debug(f"Total length of smiles output: {len(smis)}")

        if descriptor_set == "rdkit":
            descs = Descriptors.descList
            desc_names = [d[0] for d in descs]
            desc_funcs = [d[1] for d in descs]

            rows = []
            for mol_id, smi, mol in zip(ids, smis, mols):
                vals = []
                ok = True
                for f in desc_funcs:
                    try:
                        v = f(mol)
                    except Exception:
                        ok = False
                        break
                    if not np.isfinite(v):
                        ok = False
                        break
                    vals.append(v)
                if ok:
                    rows.append({"ID": mol_id, "SMILES": smi, **dict(zip(desc_names, vals))})
            df = pd.DataFrame(rows).set_index("ID")

        elif descriptor_set == "mordred":
            calc = Calculator(descriptors, ignore_3D=ignore_3D)
            df_raw = calc.pandas(mols, quiet=True)
            df_raw.index = ids

            # keep only numeric, enforce finiteness row-wise
            df_num = df_raw.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
            df_num = df_num.dropna(axis=0, how="any")
            df = pd.concat([pd.Series(smis, index=ids, name="SMILES").loc[df_num.index], df_num], axis=1)
            df.index.name = "ID"

        else:
            self.logger.error("Argument 'descriptor_set' not valid. This version only accepts 'rdkit' or 'mordred'.")
            raise ValueError("Descriptor set not supported")


        if drop_constant:
            self.logger.debug("Dropping columns with constant values")
            const_cols = [c for c in df.columns if c != "SMILES" and df[c].nunique() <= 1]
            if const_cols:
                df = df.drop(columns=const_cols, errors="ignore")

        if descriptor_set == "rdkit" and drop_fragments:
            self.logger.debug("Dropping fragment columns")
            frag_cols = [c for c in df.columns if c.startswith("fr_")]
            if frag_cols:
                df = df.drop(columns=frag_cols, errors="ignore")

        if drop_classes:
            self.logger.debug("Dropping classification-type columns (non-numeric or binary)")

            # Identify non-numeric columns (excluding SMILES)
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            non_numeric_cols = [c for c in non_numeric_cols if c != "SMILES"]

            # Identify binary (2-unique-value) numeric columns
            binary_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if df[c].nunique(dropna=True) == 2
            ]

            class_cols = non_numeric_cols + binary_cols

            if class_cols:
                self.logger.debug(f"Dropping columns: {class_cols}")
                df = df.drop(columns=class_cols, errors="ignore")

        return df

    def trainMultiTargetRFModels(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        rf_regressor_class,
        hyper_params: dict =  {
                "n_estimators": [400, 500],
                "max_features": ["sqrt"],
                "max_depth": [25, 50, 75, 100],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [2, 4, 8],
            },
        output_csv: str = "total_performance.csv",
        existing_performance_csv: str = None,
        n_resamples: int = 10,
        test_size: float = 0.3,
        cv_splits: int = 5,
        batch_size: int = 1,
        random_seed: int = 1,
        skip_existing: bool = True,
        save_models: bool = False,
        save_path: str = "./",
        log_level=logging.DEBUG
    ) -> pd.DataFrame:
        """
        Train Random Forest models for multiple target columns using logging.
        """

        # Set up empty dataframe to save performances
        total_performance_df = pd.DataFrame()

        completed_targets =  set()

        self.logger.debug("Existing performance CSV path: existing_performance_csv")
        existing_performance_csv = save_path / existing_performance_csv 

        # Load existing performance CSV
        if existing_performance_csv and Path(existing_performance_csv).exists():
            try:
                # Reading in existing performance CSV
                total_performance_df = pd.read_csv(existing_performance_csv, index_col=0)
                completed_targets = {str(c).strip() for c in total_performance_df.index}                
                self.logger.info(f"Loaded existing performance data from {existing_performance_csv}")
            except Exception as e:
                self.logger.warning(f"Could not load existing performance CSV: {e}")

        # Align indices
        common_indices = features_df.index.intersection(targets_df.index)
        if len(common_indices) == 0:
            self.logger.error("No common indices found between features_df and targets_df")
            raise ValueError("No common indices found between features_df and targets_df")

        features_df = features_df.loc[common_indices]
        self.logger.debug(f"Features DF: {features_df}")

        targets_df = targets_df.loc[common_indices]
        self.logger.debug(f"Targets DF: {features_df}")

        self.logger.info(f"Training models for {len(targets_df.columns)} target columns")
        self.logger.info(f"Using {len(common_indices)} samples with {len(features_df.columns)} features")

        # Loop through each target
        for i, target_column in enumerate(targets_df.columns):
            self.logger.debug(f"Predicting target_column...")
            target_column = str(target_column).strip()

            if skip_existing and target_column in completed_targets:
                self.logger.info(f"Skipping {target_column} ({i+1}/{len(targets_df.columns)})... already processed")
                continue

            exit

            if target_column.upper() == "SMILES":
                self.logger.info(f"Skipping {target_column}: SMILES column detected")
                continue

            self.logger.info(f"Processing target: {target_column} ({i+1}/{len(targets_df.columns)})")

            current_target = targets_df[target_column]
            combined_data = pd.concat([features_df, current_target], axis=1, join="inner")

                # Pull out the target as a Series
            y = combined_data[target_column]
            # Drop rows where target is missing
            combined_data = combined_data.loc[y.notna()].copy()

            self.logger.info(f"  After removing non-finite values: {len(combined_data)} samples")
            if y.nunique() < 2:
                self.logger.warning(f"  Skipping {target_column}: <2 unique values")
                continue

            if len(combined_data) < 10:
                self.logger.warning(f"  Skipping {target_column}: too few samples ({len(combined_data)})")
                continue

            try:
                rf_model = rf_regressor_class(
                    cv_function=KFold,
                    hp_search_function=GridSearchCV,
                    cv_kwargs={"n_splits": cv_splits, "shuffle": True, "random_state": random_seed},
                    hp_search_kwargs={"cv": cv_splits, "scoring": "neg_mean_squared_error"},
                    log_level=log_level,
                    random_seed=random_seed
                )

                final_model, best_params, performance_dict, feat_importance_df = rf_model.trainRFRegressor(
                    n_resamples=n_resamples,
                    data=combined_data,
                    target_column=target_column,
                    hyperparameters=hyper_params,
                    test_size=test_size,
                    save_interval_models=False,
                    save_path=save_path,
                    save_final_model=save_models,
                    plot_feat_importance=False,
                    batch_size=batch_size,
                    n_jobs=1,
                    final_rf_seed=random_seed
                )

                # Convert dict → DataFrame
                perf_df = pd.DataFrame([performance_dict], index=[target_column])

                # Update in-memory results
                total_performance_df = pd.concat([total_performance_df, perf_df])

                save_path = Path(save_path)
                output_csv = save_path / output_csv

                # Append to CSV under save_path
                if not output_csv.exists():
                    perf_df.to_csv(output_csv, mode="w")
                else:
                    perf_df.to_csv(output_csv, mode="a", header=False)

                self.logger.info(
                    f"  Completed {target_column} - R²: {performance_dict.get('test_r2_mean', 'N/A')}, "
                    f"RMSE: {performance_dict.get('test_rmse_mean', 'N/A')} → saved to {output_csv}"
                )

            except Exception as e:
                self.logger.error(f"  Error training model for {target_column}: {str(e)}")
                continue

        self.logger.info(f"Completed training for {len(total_performance_df)} targets")
        self.logger.info(f"Results saved to: {output_csv}")

        return total_performance_df

    def trainSingleTargetRFModel(
        self,
        data: pd.DataFrame,
        target_column: str,
        rf_regressor_class,
        hyper_params: dict =  {
                "n_estimators": [400, 500],
                "max_features": ["sqrt"],
                "max_depth": [25, 50, 75, 100],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [2, 4, 8],
            },
        n_resamples: int = 10,
        test_size: float = 0.3,
        cv_splits: int = 5,
        batch_size: int = 1,
        random_seed: int = 1,
        save_models: bool = False,
        save_path: str = "./",
        log_level=logging.DEBUG
    ) :
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)


        rf_model = rf_regressor_class(
            cv_function=KFold,
            hp_search_function=GridSearchCV,
            cv_kwargs={"n_splits": cv_splits, "shuffle": True, "random_state": random_seed},
            hp_search_kwargs={"cv": cv_splits, "scoring": "neg_mean_squared_error"},
            log_level=log_level,
            random_seed=random_seed
        )

        final_model, best_params, performance_dict, feat_importance_df = rf_model.trainRFRegressor(
            n_resamples=n_resamples,
            data=data,
            target_column=target_column,
            hyperparameters=hyper_params,
            test_size=test_size,
            save_interval_models=False,
            save_path=save_path,
            save_final_model=save_models,
            plot_feat_importance=False,
            batch_size=batch_size,
            n_jobs=1,
            final_rf_seed=random_seed
        )

        joblib.dump(final_model, Path(save_path / f"{target_column}_RF_model.pkl"))

        with open(Path(save_path / f"{target_column}_best_params.json"), "w") as f:
            json.dump(best_params, f, indent=4)

        with open(Path(save_path / f"{target_column}_performance_dict.json"), "w") as f:
            json.dump(performance_dict, f, indent=4)

        feat_importance_df.to_csv(Path(save_path / f"{target_column}_feature_importance.csv"), index_label="Feature")

        return final_model, best_params, performance_dict, feat_importance_df


    def runWorkflow(
            self,
            smiles_col: str, 
            max_len: int, 
            batch_size: int,
            test_frac: float,
            model: str,
            save_embeddings_path: Path=None
        ):

        model = model.lower()
        
        # Step 1: Generate embeddings if not already done
        if self.unembedded_df is not None and self.embedded_df is None:
            self.logger.info("No embedded_df provided. Generating embeddings...")

            if model == "chemberta":
                self.embedded_df = self.embedSMILESChemBERTa(
                    smiles_col=smiles_col,
                    max_len=max_len,
                    batch_size=batch_size,
                    save_path = save_embeddings_path
                )
            
            elif model == "molformer":
                self.embedded_df = self.embedSMILESMolFormer(
                    smiles_col=smiles_col,
                    batch_size=batch_size,
                    save_path = save_embeddings_path
                )
            else:
                self.logger.error(f"Unknown model type: '{model}'")
                raise ValueError("Model must be 'chemberta' or 'molformer'")
        
        # Step 2: Check if embeddings were successfully created
        if self.embedded_df is None or self.embedded_df.empty:
            self.logger.error("Failed to generate embeddings. `self.embedded_df` is None or empty.")
            raise RuntimeError("Embeddings not available for training.")

        # Step 3: Split into train and test
        self.logger.info(f"Running train-test split with test fraction = {test_frac}")
        self.train_df, self.test_df = train_test_split(
            self.embedded_df,
            test_size=test_frac,
            random_state=self.rng()
        )
        
        self.logger.debug("Train set preview:\n%s", self.train_df.head())
        self.logger.debug("Test set preview:\n%s", self.test_df.head())