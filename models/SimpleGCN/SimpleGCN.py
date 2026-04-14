from psutil import pids
import torch
import torch.nn as nn
import torch.nn.functional as F 
from models.SimpleGCN import GCNLayer

class SimpleGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers=3, layer_type="LightGATLayer", dropout=0.1):
        super(SimpleGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Embeddings 
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Create a list of modular layers
        if layer_type == "LightGAT":
            self.layers = nn.ModuleList([GCNLayer.LightGATLayer(dim=embedding_dim, dropout=self.dropout) for _ in range(num_layers)])
        elif layer_type == "LightGCN":
            self.layers  = nn.ModuleList([GCNLayer.LightGCNLayer(dropout=dropout)         for _ in range(num_layers)])
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

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

        u_init = self.user_embedding.weight[user_indices]
        i_init = self.item_embedding.weight[item_indices]
        
        # Predict rating via dot product
        return (u_final * i_final).sum(dim=-1), u_init, i_init
    
    def predict_ratings(self, sids, pids, edge_index, weights):
            """
            Predicts ratings for specific user-item pairs.
            
            Args:
                sids: List or array of user indices
                pids: List or array of item indices
                edge_index: The graph structure (required for GCN propagation)
                weights: The edge weights
            """
            self.eval()
            device = next(self.parameters()).device  # Get the device the model is on
            
            with torch.no_grad():
                # 1. Convert inputs to tensors on the correct device
                s_tensor = torch.as_tensor(sids, dtype=torch.long, device=device)
                p_tensor = torch.as_tensor(pids, dtype=torch.long, device=device)
                
                # 2. Call forward with the EXACT number of arguments it expects
                # forward(user_indices, item_indices, edge_index, weights)
                out, _, _ = self.forward(s_tensor, p_tensor, edge_index, weights)
                
                return out.cpu().numpy()

    def fit(
        self,
        edge_index,
        weights,
        targets,
        epochs=5000,
        lr=1e-3,
        lambda_reg=1e-4,
        loss_fn=None,
        val_fn=None,           # callable () -> scalar, used for early stopping + scheduling
        log_every=100,
        # scheduler
        scheduler_patience=50,
        scheduler_factor=0.5,
        min_lr=1e-6,
        # early stopping
        early_stop_patience=100,
        min_delta=1e-4,
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=scheduler_patience,
            factor=scheduler_factor,
            min_lr=min_lr,
        )

        user_indices = edge_index[0]
        item_indices = edge_index[1]

        best_val = float("inf")
        best_weights = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            preds, u_init, i_init = self(user_indices, item_indices, edge_index, weights)
            task_loss = loss_fn(preds, targets)
            l2_reg = lambda_reg * (u_init.norm(2) ** 2 + i_init.norm(2) ** 2)
            loss = task_loss + l2_reg

            loss.backward()
            optimizer.step()

            # use val_fn for scheduling/early stopping, fall back to train loss
            monitor = val_fn() if val_fn is not None else task_loss.item()
            scheduler.step(monitor)

            if monitor < best_val - min_delta:
                best_val = monitor
                best_weights = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch % log_every == 0:
                print(
                    f"Epoch {epoch:>5} | Loss: {loss.item():.4f} | Task: {task_loss.item():.4f}"
                    f" | Monitor: {monitor:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
                    f" | No improvement: {epochs_without_improvement}/{early_stop_patience}"
                )

            if epochs_without_improvement >= early_stop_patience:
                print(f"Early stopping at epoch {epoch} | Best monitor: {best_val:.4f}")
                break

        # restore best weights
        if best_weights is not None:
            self.load_state_dict({k: v.to(next(self.parameters()).device) for k, v in best_weights.items()})
            print(f"Restored best weights with monitor: {best_val:.4f}")

    