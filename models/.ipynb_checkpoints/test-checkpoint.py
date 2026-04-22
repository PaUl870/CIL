import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


class ColdStartALS(nn.Module):
    """
    Trains standard ALS on warm users only.
    Cold user embedding at inference = mean(item_emb[wishlisted items]).
    Item embeddings learned from warm users transfer to cold users — shared space.
    """
    def __init__(self, num_users, num_items, embedding_dim=64, lambda_reg=0.1):
        super().__init__()
        self.k          = embedding_dim
        self.lambda_reg = lambda_reg

        self.user_emb  = nn.Parameter(torch.randn(num_users, embedding_dim) * 0.01)
        self.item_emb  = nn.Parameter(torch.randn(num_items, embedding_dim) * 0.01)
        self.user_bias = nn.Parameter(torch.zeros(num_users))
        self.item_bias = nn.Parameter(torch.zeros(num_items))

        self.register_buffer("global_mean", torch.tensor(0.0))
        self.register_buffer("wl_ids_all",  torch.zeros(1, 1, dtype=torch.long))
        self.register_buffer("wl_mask_all", torch.zeros(1, 1))

    def forward(self, user_indices, item_indices):
        u   = self.user_emb[user_indices]
        i   = self.item_emb[item_indices]
        dot = (u * i).sum(dim=-1)
        return self.global_mean + self.user_bias[user_indices] + self.item_bias[item_indices] + dot

    def _cold_user_emb(self, sids):
        """Derives user embedding from wishlisted item embeddings — no user_emb lookup."""
        wl_ids  = self.wl_ids_all[sids]                                   # (B, W)
        wl_mask = self.wl_mask_all[sids]                                  # (B, W)
        vecs    = self.item_emb[wl_ids]                                   # (B, W, D)
        mask    = wl_mask.unsqueeze(-1)                                   # (B, W, 1)
        return (vecs * mask).sum(1) / mask.sum(1).clamp(min=1)            # (B, D)

    @torch.no_grad()
    def _solve_cold_user_emb(self, wishlist_df, cold_users, implicit_rating=4.0):
        """
        One closed-form ALS step for cold users using wishlist as implicit feedback.
        Treats each wishlisted item as a rating of `implicit_rating`.
        This is identical to _solve_step but only over cold users.
        """
        device = self.item_emb.device
        k      = self.k
    
        cold_list  = sorted(cold_users)
        num_cold   = len(cold_list)
        global_idx = torch.tensor(cold_list, dtype=torch.long, device=device)
    
        # Build (edge, cold_local_idx, item_idx) from wishlist
        local_map  = {sid: i for i, sid in enumerate(cold_list)}
        rows, cols = [], []
        for sid, pid in zip(wishlist_df["sid"], wishlist_df["pid"]):
            sid, pid = int(sid), int(pid)
            if sid in local_map and pid < self.item_emb.shape[0]:
                rows.append(local_map[sid])
                cols.append(pid)
    
        if not rows:
            return
    
        local_u = torch.tensor(rows, dtype=torch.long,  device=device)
        i_idx   = torch.tensor(cols, dtype=torch.long,  device=device)
        targets = torch.full((len(rows),), implicit_rating - self.global_mean.item(),
                             dtype=torch.float, device=device)
        # subtract item bias from target so solve is consistent with warm ALS
        targets = targets - self.item_bias[i_idx]
    
        cold_bias, cold_emb = self._solve_step(
            self.item_emb, i_idx, local_u, targets, num_cold
        )
    
        self.user_emb.data[global_idx]  = cold_emb
        self.user_bias.data[global_idx] = cold_bias
    
    @torch.no_grad()
    def predict_ratings(self, sids, pids, wishlist_lookup=None, batch_size=2048):
        """After fit_als + solve_cold_user_emb, cold users have proper embeddings."""
        self.eval()
        device = self.item_emb.device
        sids   = np.asarray(sids, dtype=np.int64)
        pids   = np.asarray(pids, dtype=np.int64)
        preds  = []
        for start in range(0, len(sids), batch_size):
            s = torch.as_tensor(sids[start:start+batch_size], dtype=torch.long, device=device)
            p = torch.as_tensor(pids[start:start+batch_size], dtype=torch.long, device=device)
            # Now just standard ALS forward — cold users have real embeddings
            score = self.global_mean + self.user_bias[s] + self.item_bias[p] \
                    + (self.user_emb[s] * self.item_emb[p]).sum(-1)
            preds.append(score.cpu().numpy())
        return np.clip(np.concatenate(preds), 1.0, 5.0)

    def _precompute_wishlist_tensors(self, num_users, wishlist_df, device, max_wishlist=50):
        wl_ids  = torch.zeros(num_users, max_wishlist, dtype=torch.long, device=device)
        wl_mask = torch.zeros(num_users, max_wishlist,                   device=device)
        lookup  = defaultdict(list)
        for sid, pid in zip(wishlist_df["sid"], wishlist_df["pid"]):
            lookup[int(sid)].append(int(pid))
        for sid, items in lookup.items():
            if sid >= num_users: continue
            items = items[:max_wishlist]
            wl_ids[sid,  :len(items)] = torch.tensor(items, device=device)
            wl_mask[sid, :len(items)] = 1.0
        self.wl_ids_all  = wl_ids
        self.wl_mask_all = wl_mask

    @torch.no_grad()
    def fit_als(self, train_df, wishlist_df, cold_users, num_users,
                epochs=50, val_fn=None, log_every=5):
        device = self.item_emb.device
    
        self._precompute_wishlist_tensors(num_users, wishlist_df, device)
    
        # --- compact-remap warm users ---
        warm_edges   = train_df[~train_df["sid"].isin(cold_users)].reset_index(drop=True)
        warm_sids    = np.array(sorted(warm_edges["sid"].unique()), dtype=np.int64)
        num_warm     = len(warm_sids)
        sid_to_local = {sid: i for i, sid in enumerate(warm_sids)}
    
        local_u = torch.tensor(
            warm_edges["sid"].map(sid_to_local).values, dtype=torch.long, device=device
        )
        i_idx   = torch.tensor(warm_edges["pid"].values,    dtype=torch.long,  device=device)
        weights = torch.tensor(warm_edges["rating"].values, dtype=torch.float, device=device)
    
        self.global_mean.fill_(weights.mean())
        num_items = int(self.item_emb.shape[0])
    
        # local user emb/bias — only (num_warm, k), not (num_users, k)
        local_user_emb  = self.user_emb[warm_sids].clone()
        local_user_bias = self.user_bias[warm_sids].clone()
    
        print(f"ALS: {num_warm} warm users | {num_items} items | {len(weights)} edges")
        best_val, best_state = float("inf"), None
    
        for epoch in range(epochs):
            # Update warm user embeddings
            res_u = weights - self.global_mean - self.item_bias[i_idx]
            local_user_bias, local_user_emb = self._solve_step(
                self.item_emb, i_idx, local_u, res_u, num_warm
            )
    
            # Write back to full parameter tensors
            self.user_emb.data[warm_sids]  = local_user_emb
            self.user_bias.data[warm_sids] = local_user_bias
    
            # Update item embeddings
            res_i = weights - self.global_mean - local_user_bias[local_u]
            self.item_bias.data, self.item_emb.data = self._solve_step(
                local_user_emb, local_u, i_idx, res_i, num_items
            )
    
            if epoch % log_every == 0:
                monitor = val_fn() if val_fn is not None else float("nan")
                print(f"[ColdALS] Epoch {epoch:>4} | Monitor: {monitor:.4f}")
                if val_fn is not None and monitor < best_val:
                    best_val   = monitor
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
    
        if best_state is not None:
            self.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    @torch.no_grad()
    def _solve_step(self, known_emb, known_idx, solve_idx, targets, num_solve):
        device = known_emb.device
        k      = self.k
        ones   = torch.ones(known_emb.shape[0], 1, device=device)
        V      = torch.cat([ones, known_emb], dim=1)[known_idx]

        reg_mat      = torch.eye(k + 1, device=device) * self.lambda_reg
        reg_mat[0,0] = 0.01
        A = reg_mat.unsqueeze(0).repeat(num_solve, 1, 1)
        A.scatter_add_(0, solve_idx.view(-1,1,1).expand(-1, k+1, k+1), V.unsqueeze(2) * V.unsqueeze(1))

        b = torch.zeros(num_solve, k + 1, device=device)
        b.scatter_add_(0, solve_idx.unsqueeze(1).expand(-1, k+1), V * targets.unsqueeze(1))

        sol = torch.linalg.solve(A, b.unsqueeze(2)).squeeze(2)
        return sol[:, 0], sol[:, 1:]