import torch
import torch.nn as nn
import torch.nn.functional as F

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, k: int):
        """
        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            k: Latent factors 
        """
        super().__init__()
        self.user_emb = nn.Embedding(num_users, k)
        self.item_emb = nn.Embedding(num_items, k)
        
        # Bias terms are crucial for explicit ratings (1-5 stars)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Initialization
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_indices, item_indices):
        u_e = self.user_emb(user_indices)
        i_e = self.item_emb(item_indices)
        
        # The core ALS-style interaction: Dot Product
        dot = (u_e * i_e).sum(dim=-1)
        
        # Add biases to capture baseline popularity and user optimism
        u_b = self.user_bias(user_indices).squeeze()
        i_b = self.item_bias(item_indices).squeeze()
        
        return dot + u_b + i_b + self.global_bias