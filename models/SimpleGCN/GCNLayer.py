import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, emb_dim: int, dropout: float = 0.1):
        """
        Standard GCN layer for bipartite user-item graphs.

        Args:
            emb_dim:  Embedding dimension for users and items.
            dropout:  Dropout probability applied to edges and output embeddings.
        """
        super(GCNLayer, self).__init__()
        self.dropout = dropout
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
        # ------------------------------------------------------------------ #
        # 1. Edge dropout (training only)
        #    Randomly masks entire edges, preventing over-reliance on specific
        #    interactions and acting as graph-level regularization.
        # ------------------------------------------------------------------ #
        if self.training and self.dropout > 0:
            mask = torch.rand(weights.size(0), device=weights.device) > self.dropout
            edge_index = edge_index[:, mask]
            weights = weights[mask]

        user_idx, item_idx = edge_index[0], edge_index[1]
        num_users, num_items = u_emb.size(0), i_emb.size(0)

        # ------------------------------------------------------------------ #
        # 2. Symmetric normalization  D^{-1/2} A D^{-1/2}
        # ------------------------------------------------------------------ #
        user_deg = torch.bincount(user_idx, minlength=num_users).float().clamp(min=1)
        item_deg = torch.bincount(item_idx, minlength=num_items).float().clamp(min=1)

        norm = 1.0 / (user_deg[user_idx].sqrt() * item_deg[item_idx].sqrt())
        edge_weights = (weights * norm).unsqueeze(1)  # [num_edges, 1]

        # ------------------------------------------------------------------ #
        # 3. Weighted message aggregation
        # ------------------------------------------------------------------ #
        new_u_msg = torch.zeros_like(u_emb)
        new_u_msg.index_add_(0, user_idx, i_emb[item_idx] * edge_weights)

        new_i_msg = torch.zeros_like(i_emb)
        new_i_msg.index_add_(0, item_idx, u_emb[user_idx] * edge_weights)

        # ------------------------------------------------------------------ #
        # 4. Self-connection  (I + A) formulation
        # ------------------------------------------------------------------ #
        new_u_msg = new_u_msg + u_emb
        new_i_msg = new_i_msg + i_emb

        # ------------------------------------------------------------------ #
        # 5. Linear transform + activation + embedding dropout
        # ------------------------------------------------------------------ #
        new_u_emb = F.dropout(
            F.relu(self.W_u(new_u_msg)), p=self.dropout, training=self.training
        )
        new_i_emb = F.dropout(
            F.relu(self.W_i(new_i_msg)), p=self.dropout, training=self.training
        )

        return new_u_emb, new_i_emb


class LightGCNLayer(nn.Module):
    """
    LightGCN layer (He et al., 2020).
    Removes the linear transformation, activation, and self-loop from GCN,
    keeping only the weighted neighbourhood aggregation.

    For LightGCN, only edge dropout is appropriate — there are no
    activations or linear layers to apply embedding dropout to.
    """

    def __init__(self, dropout: float = 0.1):
        super(LightGCNLayer, self).__init__()
        self.dropout = dropout

    def forward(
        self,
        u_emb: torch.Tensor,
        i_emb: torch.Tensor,
        edge_index: torch.Tensor,
        weights: torch.Tensor,
    ):
        if self.training and self.dropout > 0:
            mask = torch.rand(weights.size(0), device=weights.device) > self.dropout
            edge_index = edge_index[:, mask]
            weights = weights[mask]

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