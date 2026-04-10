import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalGCNLayer(nn.Module):
    """
    e(l)_u = σ( Σ_{(v,r)∈N(u)} 1/√(|Nu||Nv|) * W(l) * φ(e(l-1)_v, e(l-1)_r) )
    φ(e_v, e_r) = element-wise product between item and relation embedding
    e(l)_r = W(l)_rel * e(l-1)_r

    Two relation types:
        0 → rating   (weights in [0.2, 1.0])
        1 → wishlist (weights in {0, 1})
    """
    def __init__(self, emb_dim, dropout=0.1):
        super().__init__()
        self.W_u   = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_i   = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_rel = nn.Linear(emb_dim, emb_dim, bias=False)
        self.dropout = dropout

    def _weighted_dropout(self, edge_index, weights):
        if self.training and self.dropout > 0:
            drop_prob = self.dropout * (1.0 - weights)  # high weight → less likely dropped
            mask = torch.rand(weights.size(0), device=weights.device) > drop_prob
            return edge_index[:, mask], weights[mask]
        return edge_index, weights

    def forward(self, u_emb, i_emb, r_emb, edge_t0_index, edge_t0_weights, edge_t1_index, edge_t1_weights):
        """
        Args:
            u_emb:            [num_users, emb_dim]
            i_emb:            [num_items, emb_dim]
            r_emb:            [2, emb_dim]  — r_emb[0]=rating, r_emb[1]=wishlist
            edge_t0_index:    [2, E0]  rating edges
            edge_t0_weights:  [E0]     rating weights in [0.2, 1.0]
            edge_t1_index:    [2, E1]  wishlist edges
            edge_t1_weights:  [E1]     wishlist weights in {0, 1}
        Returns:
            new_u_emb, new_i_emb, new_r_emb
        """
        num_users, num_items = u_emb.size(0), i_emb.size(0)

        edge_t0_index, edge_t0_weights = self._weighted_dropout(edge_t0_index, edge_t0_weights)
        edge_t1_index, edge_t1_weights = self._weighted_dropout(edge_t1_index, edge_t1_weights)

        # ------------------------------------------------------------------ #
        # 1. Merge both edge sets, tagging each with its relation index
        # ------------------------------------------------------------------ #
        edge_index = torch.cat([edge_t0_index, edge_t1_index], dim=1)   # [2, E0+E1]
        weights    = torch.cat([edge_t0_weights, edge_t1_weights])       # [E0+E1]
        rel_ids    = torch.cat([
            torch.zeros(edge_t0_weights.size(0), dtype=torch.long, device=u_emb.device),  # 0 = rating
            torch.ones (edge_t1_weights.size(0), dtype=torch.long, device=u_emb.device),  # 1 = wishlist
        ])

        user_idx, item_idx = edge_index[0], edge_index[1]

        # ------------------------------------------------------------------ #
        # 2. Weighted symmetric normalisation across the combined graph
        # ------------------------------------------------------------------ #
        user_deg = torch.zeros(num_users, device=u_emb.device).scatter_add_(0, user_idx, weights).clamp(min=1)
        item_deg = torch.zeros(num_items, device=i_emb.device).scatter_add_(0, item_idx, weights).clamp(min=1)
        norm = (1.0 / (user_deg[user_idx].sqrt() * item_deg[item_idx].sqrt())).unsqueeze(1)

        # ------------------------------------------------------------------ #
        # 3. φ(e_v, e_r) = item ⊙ relation,  scaled by weight and norm
        # ------------------------------------------------------------------ #
        scale = (weights.unsqueeze(1) * norm)

        phi_u = i_emb[item_idx] * r_emb[rel_ids] * scale   # user ← item ⊙ relation
        phi_i = u_emb[user_idx] * r_emb[rel_ids] * scale   # item ← user ⊙ relation

        new_u_msg = torch.zeros_like(u_emb).index_add_(0, user_idx, phi_u)
        new_i_msg = torch.zeros_like(i_emb).index_add_(0, item_idx, phi_i)

        # ------------------------------------------------------------------ #
        # 4. W(l) + σ + dropout
        # ------------------------------------------------------------------ #
        new_u_emb = F.dropout(torch.sigmoid(self.W_u(new_u_msg)), p=self.dropout, training=self.training)
        new_i_emb = F.dropout(torch.sigmoid(self.W_i(new_i_msg)), p=self.dropout, training=self.training)

        # ------------------------------------------------------------------ #
        # 5. Relation update  e(l)_r = W_rel * e(l-1)_r
        # ------------------------------------------------------------------ #
        new_r_emb = self.W_rel(r_emb)

        return new_u_emb, new_i_emb, new_r_emb

