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
        self.wishlist_df = None
        
        # Dimensions for embeddings
        self.num_users = 0
        self.num_items = 0

    def load_and_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads train_ratings.csv and wishlist data, performs the coverage-safe split."""
        # --- 1. Load Ratings ---
        path_ratings = os.path.join(self.data_dir, "train_ratings.csv")
        df_ratings = pd.read_csv(path_ratings).astype({"sid": int, "pid": int})
        df_ratings = df_ratings.reset_index(drop=True)

        # Ensure coverage: one row per unique sid and pid
        sid_idx = df_ratings.groupby("sid").head(1).index
        pid_idx = df_ratings.groupby("pid").head(1).index
        core_idx = sid_idx.union(pid_idx)

        core_train = df_ratings.loc[core_idx]
        remaining = df_ratings.drop(index=core_idx)

        # Perform 75/25 split on the remaining data
        target = int(np.ceil(0.75 * len(df_ratings)))
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

        # --- 2. Load Wishlist ---
        path_wishlist = os.path.join(self.data_dir, "train_tbr.csv")
        if os.path.exists(path_wishlist):
            self.wishlist_df = pd.read_csv(path_wishlist).astype({"sid": int, "pid": int})
        else:
            print(f"Warning: Wishlist file not found at {path_wishlist}. Proceeding with empty wishlist.")
            self.wishlist_df = pd.DataFrame(columns=["sid", "pid"])

        # --- 3. Update Dimensions across both datasets ---
        max_sid_ratings = df_ratings["sid"].max()
        max_pid_ratings = df_ratings["pid"].max()
        
        max_sid_wish = self.wishlist_df["sid"].max() if not self.wishlist_df.empty else 0
        max_pid_wish = self.wishlist_df["pid"].max() if not self.wishlist_df.empty else 0

        self.num_users = max(max_sid_ratings, max_sid_wish) + 1
        self.num_items = max(max_pid_ratings, max_pid_wish) + 1
        
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
        """Converts train_df and wishlist_df into edge_index and weights for GCN models."""
        if self.train_df is None:
            raise ValueError("Data not loaded. Run load_and_split() first.")
            
        # Type 0: Ratings
        edge_index_t0 = torch.tensor(self.train_df[["sid", "pid"]].values.T, dtype=torch.long)
        # Normalize ratings to [0, 1] for propagation
        weights_t0 = torch.tensor(self.train_df["rating"].values / 5.0, dtype=torch.float)
        
        # Type 1: Wishlist
        if self.wishlist_df is not None and not self.wishlist_df.empty:
            edge_index_t1 = torch.tensor(self.wishlist_df[["sid", "pid"]].values.T, dtype=torch.long)
            # Implicit feedback: assign a uniform weight of 1.0 to all wishlist edges
            weights_t1 = torch.ones(edge_index_t1.shape[1], dtype=torch.float)
        else:
            edge_index_t1 = torch.empty((2, 0), dtype=torch.long)
            weights_t1 = torch.empty((0,), dtype=torch.float)
            
        return (
            edge_index_t0.to(device), weights_t0.to(device),
            edge_index_t1.to(device), weights_t1.to(device)
        )