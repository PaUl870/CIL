import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SimpleGCN import GCNLayer


class MultiRelGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super(MultiRelGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Separate GCN layers per edge type
        # edge type 0: e.g. ratings
        # edge type 1: e.g. wishlist / implicit
        self.layers_type0 = nn.ModuleList(
            [GCNLayer.LightGCNLayer(dropout=dropout) for _ in range(num_layers)]
        )
        self.layers_type1 = nn.ModuleList(
            [GCNLayer.LightGCNLayer(dropout=dropout) for _ in range(num_layers)]
        )

        # Learned per-type importance scalar (how much each edge type contributes)
        self.type_weights = nn.Parameter(torch.ones(2))

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def _propagate(self, layers, u_emb, i_emb, edge_index, weights):
        """Run one edge type through its GCN stack, return mean-pooled embeddings."""
        all_u = [u_emb]
        all_i = [i_emb]
        for layer in layers:
            u_emb, i_emb = layer(u_emb, i_emb, edge_index, weights)
            all_u.append(u_emb)
            all_i.append(i_emb)
        return (
            torch.stack(all_u, dim=1).mean(dim=1),
            torch.stack(all_i, dim=1).mean(dim=1),
        )

    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        edge_index_t0: torch.Tensor,  # [2, num_edges] for type 0
        weights_t0: torch.Tensor,     # [num_edges] for type 0
        edge_index_t1: torch.Tensor,  # [2, num_edges] for type 1
        weights_t1: torch.Tensor,     # [num_edges] for type 1
    ) -> torch.Tensor:

        u0 = self.user_embedding.weight
        i0 = self.item_embedding.weight

        # Apply embedding dropout once before propagation
        u0 = F.dropout(u0, p=self.dropout, training=self.training)
        i0 = F.dropout(i0, p=self.dropout, training=self.training)

        # Propagate each edge type independently
        u_t0, i_t0 = self._propagate(self.layers_type0, u0, i0, edge_index_t0, weights_t0)
        u_t1, i_t1 = self._propagate(self.layers_type1, u0, i0, edge_index_t1, weights_t1)

        # Weighted combination of the two edge types
        tw = F.softmax(self.type_weights, dim=0)
        final_u = tw[0] * u_t0 + tw[1] * u_t1
        final_i = tw[0] * i_t0 + tw[1] * i_t1

        # Extract batch and score
        u_final = final_u[user_indices]
        i_final = final_i[item_indices]
        return (u_final * i_final).sum(dim=-1)

    @torch.no_grad()
    def get_all_embeddings(self, edge_index_t0, weights_t0, edge_index_t1, weights_t1):
        u0 = self.user_embedding.weight
        i0 = self.item_embedding.weight
        u_t0, i_t0 = self._propagate(self.layers_type0, u0, i0, edge_index_t0, weights_t0)
        u_t1, i_t1 = self._propagate(self.layers_type1, u0, i0, edge_index_t1, weights_t1)
        tw = F.softmax(self.type_weights, dim=0)
        return tw[0] * u_t0 + tw[1] * u_t1, tw[0] * i_t0 + tw[1] * i_t1