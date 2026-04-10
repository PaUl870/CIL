import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SimpleGCN.RelationalGCNLayer import RelationalGCNLayer


class MultiRelGCN(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        self.dropout = dropout

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

            # Two relation embeddings: 0=rating, 1=wishlist
        self.relation_embedding = nn.Embedding(2, embedding_dim)
        nn.init.normal_(self.relation_embedding.weight, std=0.01)
        self.layers = nn.ModuleList([
            RelationalGCNLayer(embedding_dim, dropout) for _ in range(num_layers)
        ])


    def _propagate(self, edge_t0_index, edge_t0_weights, edge_t1_index, edge_t1_weights):
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight

        all_u = [u_emb]
        all_i = [i_emb]

        r_emb = self.relation_embedding.weight
        for layer in self.layers:
            u_emb, i_emb, r_emb = layer(u_emb, i_emb, r_emb, edge_t0_index, edge_t0_weights, edge_t1_index, edge_t1_weights)
            all_u.append(u_emb)
            all_i.append(i_emb)


        final_u = torch.stack(all_u, dim=1).mean(dim=1)
        final_i = torch.stack(all_i, dim=1).mean(dim=1)
        return final_u, final_i

    def forward(
        self,
        user_indices,       # [B]
        item_indices,       # [B]
        edge_t0_index,      # [2, E0]  rating edges
        edge_t0_weights,    # [E0]
        edge_t1_index,      # [2, E1]  wishlist edges
        edge_t1_weights,    # [E1]
    ):
        final_u, final_i = self._propagate(edge_t0_index, edge_t0_weights, edge_t1_index, edge_t1_weights)

        u_final = final_u[user_indices]
        i_final = final_i[item_indices]

        return (u_final * i_final).sum(dim=-1)