import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, emb_dim: int):
        """
        Standard GCN layer for bipartite user-item graphs.

        Args:
            emb_dim: Embedding dimension for users and items.
        """
        super(GCNLayer, self).__init__()
        self.W_u = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_i = nn.Linear(emb_dim, emb_dim, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_u.weight)
        nn.init.xavier_uniform_(self.W_i.weight)

    def forward(
        self,
        u_emb: torch.Tensor,
        i_emb: torch.Tensor,
        edge_index: torch.Tensor,
        weights: torch.Tensor,
    ):
        """
        Args:
            u_emb:      [num_users, emb_dim]  – current user embeddings
            i_emb:      [num_items, emb_dim]  – current item embeddings
            edge_index: [2, num_edges]        – (user_idx, item_idx) pairs
            weights:    [num_edges]           – normalized edge weights
        Returns:
            new_u_emb: [num_users, emb_dim]
            new_i_emb: [num_items, emb_dim]
        """
        user_idx, item_idx = edge_index[0], edge_index[1]
        num_users, num_items = u_emb.size(0), i_emb.size(0)

        # ------------------------------------------------------------------ #
        # 1. Symmetric normalization  D^{-1/2} A D^{-1/2}
        #    Each edge (u, i) is scaled by  1 / sqrt(deg_u * deg_i)
        # ------------------------------------------------------------------ #
        user_deg = torch.bincount(user_idx, minlength=num_users).float().clamp(min=1)
        item_deg = torch.bincount(item_idx, minlength=num_items).float().clamp(min=1)

        # Per-edge normalization factor
        norm = 1.0 / (user_deg[user_idx].sqrt() * item_deg[item_idx].sqrt())

        # Combine rating weights with structural normalization
        edge_weights = (weights * norm).unsqueeze(1)  # [num_edges, 1]

        # ------------------------------------------------------------------ #
        # 2. Weighted message aggregation
        # ------------------------------------------------------------------ #
        # Users aggregate from neighbouring items
        new_u_msg = torch.zeros_like(u_emb)
        new_u_msg.index_add_(0, user_idx, i_emb[item_idx] * edge_weights)

        # Items aggregate from neighbouring users
        new_i_msg = torch.zeros_like(i_emb)
        new_i_msg.index_add_(0, item_idx, u_emb[user_idx] * edge_weights)

        # ------------------------------------------------------------------ #
        # 3. Self-connection  (I + A) formulation
        # ------------------------------------------------------------------ #
        new_u_msg = new_u_msg + u_emb
        new_i_msg = new_i_msg + i_emb

        # ------------------------------------------------------------------ #
        # 4. Learnable linear transformation + non-linearity
        # ------------------------------------------------------------------ #
        new_u_emb = F.relu(self.W_u(new_u_msg))
        new_i_emb = F.relu(self.W_i(new_i_msg))

        return new_u_emb, new_i_emb


# ------------------------------------------------------------------ #
# Optional: LightGCN variant (no W, no activation, no self-loop)
# Preferred for collaborative filtering – simpler and often stronger.
# ------------------------------------------------------------------ #
class LightGCNLayer(nn.Module):
    """
    LightGCN layer (He et al., 2020).
    Removes the linear transformation, activation, and self-loop from GCN,
    keeping only the weighted neighbourhood aggregation.
    """

    def forward(
        self,
        u_emb: torch.Tensor,
        i_emb: torch.Tensor,
        edge_index: torch.Tensor,
        weights: torch.Tensor,
    ):
        user_idx, item_idx = edge_index[0], edge_index[1]
        num_users, num_items = u_emb.size(0), i_emb.size(0)

        user_deg = torch.bincount(user_idx, minlength=num_users).float().clamp(min=1)
        item_deg = torch.bincount(item_idx, minlength=num_items).float().clamp(min=1)

        norm = 1.0 / (user_deg[user_idx].sqrt() * item_deg[item_idx].sqrt())
        edge_weights = (weights * norm).unsqueeze(1)

        new_u_emb = torch.zeros_like(u_emb)
        new_u_emb.index_add_(0, user_idx, i_emb[item_idx] * edge_weights)

        new_i_emb = torch.zeros_like(i_emb)
        new_i_emb.index_add_(0, item_idx, u_emb[user_idx] * edge_weights)

        return new_u_emb, new_i_emb