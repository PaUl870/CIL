import torch
import torch.nn as nn

class MultiRelCompGCN(nn.Module):
    def __init__(self, num_nodes, num_rels, embed_dim):
        super(MultiRelCompGCN, self).__init__()
        
        # Base node and relation embeddings
        self.node_embeds = nn.Embedding(num_nodes, embed_dim)
        self.rel_embeds = nn.Embedding(num_rels, embed_dim)
        
        # The Relation Encoder: This interprets the weight BASED on the relation type
        # We use a slightly wider hidden layer to handle different types of weights
        self.rel_encoder = nn.Sequential(
            nn.Linear(embed_dim + 1, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Decoder to get the weight back
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, head_ids, rel_ids, tail_ids, weights):
        # 1. Look up semantic base
        h_u = self.node_embeds(head_ids)
        h_v = self.node_embeds(tail_ids)
        z_r = self.rel_embeds(rel_ids) 
        
        # 2. Inject weight into the relation vector
        # For 'Rating', weight might be 0.9. For 'Wishlist', weight is 1.0.
        rel_combined = torch.cat([z_r, weights.unsqueeze(1)], dim=1)
        h_r_dynamic = self.rel_encoder(rel_combined)
        
        # 3. Composition (Geometric interaction)
        # subtraction allows the weight to shift the head toward the tail
        phi = h_u - h_r_dynamic
        
        # 4. Decode the scalar back
        # The model uses the specific relation context to know if it's decoding a rating or a flag
        out = torch.cat([phi, h_r_dynamic, h_v], dim=1)
        return self.decoder(out).squeeze()