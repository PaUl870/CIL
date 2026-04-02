import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from typing import Tuple, Callable

class DataLoader:
    def __init__(self, data_dir: str, seed: int = 42):
        self.data_dir = data_dir
        self.seed = seed
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        
        # Dimensions for embeddings
        self.num_users = 0
        self.num_items = 0

    def load_and_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads train_ratings.csv and performs the coverage-safe split."""
        path = os.path.join(self.data_dir, "train_ratings.csv")
        df = pd.read_csv(path).astype({"sid": int, "pid": int})
        df = df.reset_index(drop=True)

        # 1. Ensure coverage: one row per unique sid and pid
        sid_idx = df.groupby("sid").head(1).index
        pid_idx = df.groupby("pid").head(1).index
        core_idx = sid_idx.union(pid_idx)

        core_train = df.loc[core_idx]
        remaining = df.drop(index=core_idx)

        # 2. Perform 75/25 split on the remaining data
        target = int(np.ceil(0.75 * len(df)))
        add_n = min(max(0, target - len(core_train)), len(remaining))

        if add_n > 0:
            add_train, val = train_test_split(
                remaining, train_size=add_n, random_state=self.seed, shuffle=True
            )
            self.train_df = pd.concat([core_train, add_train], ignore_index=True)
            self.valid_df = val.reset_index(drop=True)
        else:
            self.train_df = core_train.reset_index(drop=True)
            self.valid_df = remaining.reset_index(drop=True)

        # Update dimensions based on the full dataset
        self.num_users = df["sid"].max() + 1
        self.num_items = df["pid"].max() + 1
        
        return self.train_df, self.valid_df

    def get_test_df(self) -> pd.DataFrame:
        """Loads and formats the Kaggle test set."""
        path = os.path.join(self.data_dir, "test_ratings.csv")
        df = pd.read_csv(path)
        # Split 'sid_pid' string into two columns
        split_ids = df["sid_pid"].str.split("_", expand=True)
        df["sid"] = split_ids[0].astype(int)
        df["pid"] = split_ids[1].astype(int)
        self.test_df = df.drop("sid_pid", axis=1)
        return self.test_df

    def get_graph_tensors(self, device='cpu'):
        """Converts train_df into edge_index and weights for GCN models."""
        if self.train_df is None:
            raise ValueError("Data not loaded. Run load_and_split() first.")
            
        edge_index = torch.tensor(self.train_df[["sid", "pid"]].values.T, dtype=torch.long)
        # Normalize ratings to [0, 1] for propagation
        edge_weights = torch.tensor(self.train_df["rating"].values / 5.0, dtype=torch.float)
        
        return edge_index.to(device), edge_weights.to(device)

    def make_submission(self, pred_fn: Callable, output_path: str):
        """Standardized submission generator."""
        sample_path = os.path.join(self.data_dir, "sample_submission.csv")
        sub_df = pd.read_csv(sample_path)
        
        # Extract IDs from the sample format
        sid_pid = sub_df["sid_pid"].str.split("_", expand=True)
        sids = sid_pid[0].astype(int).values
        pids = sid_pid[1].astype(int).values

        # Generate predictions
        sub_df["rating"] = pred_fn(sids, pids)
        sub_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")