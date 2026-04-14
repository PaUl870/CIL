import torch
def enmf_loss(user_embeddings, item_embeddings, edge_index, alpha=0.1):
    """
    Non-sampling loss implementation.
    alpha: the weight for unobserved entries (the 'missing' data).
    """
    # 1. Loss for observed interactions (Wishlist)
    # Get embeddings for the edges that actually exist
    u_idx, i_idx = edge_index[0], edge_index[1]
    pos_preds = (user_embeddings[u_idx] * item_embeddings[i_idx]).sum(dim=1)
    # We want these to be 1.0. Weight = (1 - alpha)
    pos_loss = ((1 - alpha) * (1 - pos_preds)**2).sum()

    # 2. Loss for ALL possible interactions (The efficiency trick)
    # Weight = alpha. We want all scores to be 0.0 unless they are in the wishlist.
    # We use the property: Sum( (p_u @ q_i)^2 ) = Trace( P @ (Q.T @ Q) @ P.T )
    
    # Precompute Q^T Q for all items
    QTQ = torch.matmul(item_embeddings.T, item_embeddings) # Shape: (dim, dim)
    
    # Calculate p_u^T (Q^T Q) p_u for each user in the batch
    # This represents the sum of squared predictions for ALL items for these users
    all_loss = torch.matmul(user_embeddings, QTQ) # (num_users, dim)
    all_loss = (all_loss * user_embeddings).sum() # Sum across users and dimensions
    
    return pos_loss + (alpha * all_loss)