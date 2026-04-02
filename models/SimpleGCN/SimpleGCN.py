import torch
import torch.nn as nn
import torch.nn.functional as F 
from models.SimpleGCN import GCNLayer

class SimpleGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers=3, dropout=0.1):
        super(SimpleGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embeddings 
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Create a list of modular layers
        self.layers = nn.ModuleList([GCNLayer.LightGCNLayer() for _ in range(num_layers)])
        
        # Initialization
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_indices, item_indices, edge_index, weights):
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        
        all_u = [u_emb]
        all_i = [i_emb]
        
        for layer in self.layers:
            # Propagate through the layer
            u_msg, i_msg = layer(u_emb, i_emb, edge_index, weights)
            
            # Update embeddings for the next layer
            u_emb = u_msg
            i_emb = i_msg

            u_emb = F.dropout(u_emb, p=self.dropout, training=self.training)
            i_emb = F.dropout(i_emb, p=self.dropout, training=self.training)
            
            all_u.append(u_emb)
            all_i.append(i_emb)
        
        # Final embedding is the average of all layers
        # This helps prevent over-smoothing in deep GCNs
        final_u_embeddings = torch.stack(all_u, dim=1).mean(dim=1)
        final_i_embeddings = torch.stack(all_i, dim=1).mean(dim=1)
        
        # Extract specific embeddings for the current batch
        u_final = final_u_embeddings[user_indices]
        i_final = final_i_embeddings[item_indices]
        
        # Predict rating via dot product
        return (u_final * i_final).sum(dim=-1)