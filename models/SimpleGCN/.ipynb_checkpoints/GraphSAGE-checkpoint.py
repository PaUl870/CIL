import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SimpleGCN import GCNLayer
import math

class GraphSAGE(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_mlp_head: bool = False,
        use_bilinear_head: bool = False,
        mlp_hidden_dims: list[int] = (256, 64),
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Inductive components
        self.layers = nn.ModuleList([
            GCNLayer.SAGELayer(embedding_dim, dropout) for _ in range(num_layers)
        ])
        
        # Subgraph weighting components
        self.attn_layer = RatingAttention(embedding_dim, 6)
        self.rel_projections = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(6)
        ])

        # Scorer
        if use_mlp_head:
            self.score_head = PredictionMLP(embedding_dim, list(mlp_hidden_dims))
        elif use_bilinear_head:
            self.score_head = BilinearScorer(embedding_dim)
        else:
            self.score_head = None

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def _score(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        if self.score_head is not None:
            return self.score_head(u, i)
        return (u * i).sum(dim=-1)

    def forward(self, user_indices, item_indices, edge_index, edge_index_t1, weights):
        # 1. Split into Subgraphs (Rating 1-5 + Wishlist)
        subgraphs = []
        unique_ratings = torch.unique(weights)
        for r in unique_ratings:
            subgraphs.append({
                "edge_index": edge_index[:, weights == r],
                "proj_idx": int(r.item()) - 1
            })

        if edge_index_t1 is not None and edge_index_t1.size(1) > 0:
            subgraphs.append({
                "edge_index": edge_index_t1,
                "proj_idx": 5 # Wishlist index
            })

        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        
        # Track hidden states for layer-aggregation (optional, typical for GCNs)
        all_u, all_i = [u_emb], [i_emb]

        for layer in self.layers:
            sub_msgs_u, sub_msgs_i = [], []
            u_counts_list, i_counts_list = [], []

            for graph in subgraphs:
                edges = graph["edge_index"]
                
                # Degree calculation for attention masking
                u_deg = torch.zeros(self.num_users, device=edges.device)
                i_deg = torch.zeros(self.num_items, device=edges.device)
                u_deg.index_add_(0, edges[0], torch.ones(edges.size(1), device=edges.device))
                i_deg.index_add_(0, edges[1], torch.ones(edges.size(1), device=edges.device))
                u_counts_list.append(u_deg)
                i_counts_list.append(i_deg)

                # SAGE Message Passing
                u_msg, i_msg = layer(u_emb, i_emb, edges)

                # Project based on relation type
                p_idx = graph["proj_idx"]
                u_msg = self.rel_projections[p_idx](u_msg)
                i_msg = self.rel_projections[p_idx](i_msg)

                sub_msgs_u.append(u_msg)
                sub_msgs_i.append(i_msg)

            # 2. Multi-Relational Attention Aggregation
            u_subgraph_counts = torch.stack(u_counts_list, dim=1)
            i_subgraph_counts = torch.stack(i_counts_list, dim=1)

            u_emb = self.attn_layer(u_emb, sub_msgs_u, u_subgraph_counts)
            i_emb = self.attn_layer(i_emb, sub_msgs_i, i_subgraph_counts)

            all_u.append(u_emb)
            all_i.append(i_emb)

        # 3. Final readout (Mean across layers)
        final_u = torch.stack(all_u, dim=1).mean(dim=1)
        final_i = torch.stack(all_i, dim=1).mean(dim=1)

        return self._score(final_u[user_indices], final_i[item_indices]), \
               self.user_embedding.weight[user_indices], \
               self.item_embedding.weight[item_indices]

    def fit(
        self,
        edge_index,
        edge_index_t1,
        weights,
        targets,
        epochs: int = 5000,
        lr: float = 1e-3,
        lambda_reg: float = 1e-4,
        loss_fn=None,
        val_fn=None,
        log_every: int = 100,
        scheduler_patience: int = 50,
        scheduler_factor: float = 0.5,
        min_lr: float = 1e-6,
        early_stop_patience: int = 100,
        min_delta: float = 1e-4,
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = loss_fn or nn.MSELoss()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=scheduler_patience,
            factor=scheduler_factor,
            min_lr=min_lr,
        )

        # Indices for the training pairs (the purchase edges)
        user_indices = edge_index[0]
        item_indices = edge_index[1]

        best_val = float("inf")
        best_weights = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            # Forward pass: 
            # Note: we use the full graph (edge_index and edge_index_t1) 
            # to compute embeddings, then score the training pairs.
            preds, u_init, i_init = self(
                user_indices, item_indices, edge_index, edge_index_t1, weights
            )
            
            task_loss = loss_fn(preds, targets)
            
            # L2 Regularization on initial embeddings to prevent overfitting
            l2_reg = lambda_reg * (
                u_init.pow(2).sum(dim=1).mean() +
                i_init.pow(2).sum(dim=1).mean()
            )            
            
            loss = task_loss + l2_reg

            loss.backward()
            optimizer.step()

            # Validation / Monitoring
            # If a val_fn is provided (e.g., checking cold-start RMSE), use it
            monitor = val_fn() if val_fn is not None else task_loss.item()
            scheduler.step(monitor)

            # Early Stopping Logic
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

        if best_weights is not None:
            device = next(self.parameters()).device
            self.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
            print(f"Restored best weights | Monitor: {best_val:.4f}")

    @torch.no_grad()
    def predict_ratings(self, sids, pids, edge_index, edge_index_t1, weights):
        """Predict ratings for specific user-item pairs."""
        self.eval()
        device = next(self.parameters()).device
    
        s_tensor = torch.as_tensor(sids, dtype=torch.long, device=device)
        p_tensor = torch.as_tensor(pids, dtype=torch.long, device=device)
    
        # Ensure weights are passed into the forward call here too
        out, _, _ = self.forward(s_tensor, p_tensor, edge_index, edge_index_t1, weights)
        return out.cpu().numpy()



class RatingAttention(nn.Module):
    def __init__(self, emb_dim: int, num_subgraphs: int):
        super().__init__()
        # We add +1 to the input dim to account for the 'count' feature
        self.norm = nn.LayerNorm(emb_dim * 2 + 1)
        self.attn_project = nn.Sequential(
            nn.Linear(emb_dim * 2 + 1, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1),
        )
        nn.init.xavier_uniform_(self.attn_project[-1].weight, gain=1.0)
        nn.init.zeros_(self.attn_project[-1].bias)
        self.last_weights = None  # To store diagnostics

    def forward(self, query_emb, subgraph_messages, subgraph_counts):
        # 1. Prepare inputs first
        stacked = torch.stack(subgraph_messages, dim=1)   # [N, K, dim]
        K = stacked.size(1)
        query_expanded = query_emb.unsqueeze(1).expand(-1, K, -1)  # [N, K, dim]
        counts_feat = subgraph_counts.unsqueeze(-1)  # [N, K, 1]
        
        # 2. Concatenate inputs
        features = torch.cat([query_expanded, stacked, counts_feat], dim=-1)
        
        # 3. Normalize and compute scores
        features = self.norm(features) 
        scores = self.attn_project(features)
        
        # 4. Mask and compute weights
        scores = scores.masked_fill(counts_feat == 0, -1e9)
        weights = F.softmax(scores, dim=1)
        
        # 5. Output
        self.last_weights = weights.detach()
        return (stacked * weights).sum(dim=1)  