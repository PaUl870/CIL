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
        user_deg = torch.zeros(num_users, device=weights.device).scatter_add_(0, user_idx, weights)
        item_deg = torch.zeros(num_items, device=weights.device).scatter_add_(0, item_idx, weights)

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

        user_deg = torch.zeros(num_users, device=weights.device).scatter_add_(0, user_idx, weights)
        item_deg = torch.zeros(num_items, device=weights.device).scatter_add_(0, item_idx, weights)

        norm = 1.0 / (user_deg[user_idx].sqrt() * item_deg[item_idx].sqrt())
        edge_weights = (weights * norm).unsqueeze(1)

        new_u_emb = torch.zeros_like(u_emb)
        new_u_emb.index_add_(0, user_idx, i_emb[item_idx] * edge_weights)

        new_i_emb = torch.zeros_like(i_emb)
        new_i_emb.index_add_(0, item_idx, u_emb[user_idx] * edge_weights)

        return new_u_emb, new_i_emb
    


class LightGATLayer(nn.Module):
    """
    Lightweight GAT layer for bipartite user-item graphs.
    Single relation (e.g. ratings only).
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.attn_projector = nn.Linear(dim * 2, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
 
    def _weighted_dropout(self, edge_index, weights):
        if self.training and self.dropout > 0:
            drop_prob = self.dropout * (1.0 - weights)
            mask = torch.rand(weights.size(0), device=weights.device) > drop_prob
            return edge_index[:, mask], weights[mask]
        return edge_index, weights
 
    def _scatter_softmax(self, scores, idx, num_nodes):
        score_max = torch.full((num_nodes,), float('-inf'), device=scores.device)
        score_max.index_reduce_(0, idx, scores, reduce='amax', include_self=True)
        exp_scores = torch.exp(scores - score_max[idx])
        exp_sum = torch.zeros(num_nodes, device=scores.device)
        exp_sum.index_add_(0, idx, exp_scores)
        return exp_scores / (exp_sum[idx] + 1e-10)
 
    def forward(self, u_emb, i_emb, edge_index, weights):
        """
        Args:
            u_emb:      [num_users, dim]
            i_emb:      [num_items, dim]
            edge_index: [2, E]
            weights:    [E]  in [0.2, 1.0]
        Returns:
            new_u_emb: [num_users, dim]
            new_i_emb: [num_items, dim]
        """
        num_users, num_items = u_emb.size(0), i_emb.size(0)
 
        edge_index, weights = self._weighted_dropout(edge_index, weights)
        user_idx, item_idx = edge_index[0], edge_index[1]
 
        # 1. Attention scores scaled by edge weight
        edge_feat = torch.cat([u_emb[user_idx], i_emb[item_idx]], dim=-1)
        raw_scores = self.leaky_relu(self.attn_projector(edge_feat).squeeze(-1)) 
 
        # 2. Scatter softmax per node neighbourhood
        u_attn = self._scatter_softmax(raw_scores, user_idx, num_users)
        i_attn = self._scatter_softmax(raw_scores, item_idx, num_items)
 
        # 3. Symmetric aggregation + self-connection
        new_u_emb = u_emb.clone()
        new_u_emb.index_add_(0, user_idx, i_emb[item_idx] * u_attn.unsqueeze(-1))
 
        new_i_emb = i_emb.clone()
        new_i_emb.index_add_(0, item_idx, u_emb[user_idx] * i_attn.unsqueeze(-1))
 
        return new_u_emb, new_i_emb