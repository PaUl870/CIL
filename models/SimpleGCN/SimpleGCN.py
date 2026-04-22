import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SimpleGCN import GCNLayer
import math


class PredictionMLP(nn.Module):
    """Optional MLP head that replaces the dot-product scorer."""

    def __init__(self, emb_dim: int, hidden_dims: list[int] = (256, 64)):
        super().__init__()
        in_dim = emb_dim * 2          # concat(u, i)
        dims = [in_dim, *hidden_dims]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: [B, dim]
            i: [B, dim]
        Returns:
            [B] scores in (0, 1)
        """
        return self.net(torch.cat([u, i], dim=-1)).squeeze(1)


class BilinearScorer(nn.Module):
    """
    Computes u^T * Q * i + bias.
    This allows for weighted interactions between specific dimensions 
    of user and item embeddings.
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        # Q matrix: [dim, dim]
        self.Q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Initialize Q as an identity-like matrix to start near a dot product
        nn.init.xavier_uniform_(self.Q.weight)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """
        u: [B, dim], i: [B, dim]
        Returns: [B] scores in (0, 1)
        """
        # (i @ Q^T) results in [B, dim]
        # Then element-wise multiply with u and sum across rows
        interaction = torch.sum(u * self.Q(i), dim=-1) + self.bias
        return interaction


class SimpleGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        num_layers: int = 3,
        layer_type: str = "LightGCN",
        dropout: float = 0.1,
        use_mlp_head: bool = False,
        use_bilinear_head: bool = False, 
        mlp_hidden_dims: list[int] = (256, 64),
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.attn_layer = RatingAttention(embedding_dim, 6)
        self.rel_projections = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(6) 
        ])


        if layer_type == "LightGAT":
            self.layers = nn.ModuleList([
                GCNLayer.LightGATLayer(dim=embedding_dim, dropout=dropout)
                for _ in range(num_layers)
            ])
        elif layer_type == "GCN":
            self.layers = nn.ModuleList([
                GCNLayer.GCNLayer(emb_dim=embedding_dim, dropout=dropout)
                for _ in range(num_layers)
            ])
        elif layer_type == "LightGCN":
            self.layers = nn.ModuleList([
                GCNLayer.LightGCNLayer(dropout=dropout)
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}") 

        # Prediction head logic
        if use_mlp_head:
            self.score_head = PredictionMLP(embedding_dim, list(mlp_hidden_dims))
        elif use_bilinear_head:
            self.score_head = BilinearScorer(embedding_dim)
        else:
            self.score_head = None # Default to dot product

        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def _score(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """Unified scoring logic."""
        if self.score_head is not None:
            return self.score_head(u, i)
        
        # Fallback to Scaled Dot Product
        return (u * i).sum(dim=-1)

    def forward(self, user_indices, item_indices, edge_index, edge_index_t1, weights):
        # 1. Create subgraphs with explicit projection indices
        subgraphs = []
        unique_ratings = torch.unique(weights)
        
        for r in unique_ratings:
            subgraphs.append({
                "edge_index": edge_index[:, weights == r],
                "proj_idx": int(r.item()) - 1  # Ratings 1-5 -> indices 0-4
            })

        if edge_index_t1 is not None and edge_index_t1.size(1) > 0:
            subgraphs.append({
                "edge_index": edge_index_t1,
                "proj_idx": 5  # Wishlist -> 6th projection (index 5)
            })

        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        all_u, all_i = [u_emb], [i_emb]

        for layer in self.layers:
            sub_msgs_u, sub_msgs_i = [], []
            u_counts_list = []
            i_counts_list = []
            for graph in subgraphs:
                edges = graph["edge_index"]
                # Count occurrences of each user/item index in this subgraph
                u_deg = torch.zeros(self.num_users, device=edges.device)
                i_deg = torch.zeros(self.num_items, device=edges.device)
                u_counts_list.append(u_deg)
                i_counts_list.append(i_deg)
                # 2. Use dummy weights for the GCN layer
                w_dummy = torch.ones(graph["edge_index"].size(1), device=edge_index.device)
                u_msg, i_msg = layer(u_emb, i_emb, graph["edge_index"], w_dummy)
                
                # 3. Retrieve the correct projection index directly from the dict
                p_idx = graph["proj_idx"]
                u_msg = self.rel_projections[p_idx](u_msg)
                i_msg = self.rel_projections[p_idx](i_msg)
                
                sub_msgs_u.append(u_msg)
                sub_msgs_i.append(i_msg)

            u_subgraph_counts = torch.stack(u_counts_list, dim=1) # [num_users, K]
            i_subgraph_counts = torch.stack(i_counts_list, dim=1) # [num_items, K]
        
            # ... inside the layer loop ...
            u_emb = self.attn_layer(u_emb, sub_msgs_u, u_subgraph_counts)
            i_emb = self.attn_layer(i_emb, sub_msgs_i, i_subgraph_counts)
            # ... (rest of the logic remains the same)

            u_emb = F.dropout(u_emb, p=self.dropout, training=self.training)
            i_emb = F.dropout(i_emb, p=self.dropout, training=self.training)

            all_u.append(u_emb)
            all_i.append(i_emb)

        final_u = torch.stack(all_u, dim=1).mean(dim=1)
        final_i = torch.stack(all_i, dim=1).mean(dim=1)

        u_final = final_u[user_indices]
        i_final = final_i[item_indices]

        u_init = self.user_embedding.weight[user_indices]
        i_init = self.item_embedding.weight[item_indices]

        prediction = self._score(u_final, i_final)  # ← uses MLP or dot product
        return prediction, u_init, i_init
        
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

    def fit(
        self,
        edge_index,
        edge_index_t1,
        weights,
        targets,
        epochs: int = 5000,
        lr: float = 1e-3,import torch
from collections import defaultdict


@torch.no_grad()
def initialize_cold_start(model, wishlist_df, cold_users, cold_items, top_k: int = 5):
    device = model.user_embedding.weight.device

    user_wishlist = defaultdict(set)
    item_wishlist = defaultdict(set)
    for sid, pid in zip(wishlist_df["sid"].values, wishlist_df["pid"].values):
        user_wishlist[int(sid)].add(int(pid))
        item_wishlist[int(pid)].add(int(sid))

    def jaccard_aggregate(cold_entities, entity_wishlist,
                          counterpart_wishlist, warm_mask_set, emb_matrix):
        updated, skipped = 0, 0
        for eid in cold_entities:
            cold_set = entity_wishlist.get(eid, set())
            if not cold_set:
                skipped += 1
                continue

            overlap_counts = defaultdict(int)
            for counterpart in cold_set:
                for warm_eid in counterpart_wishlist.get(counterpart, set()):
                    if warm_eid in warm_mask_set:
                        overlap_counts[warm_eid] += 1

            if not overlap_counts:
                skipped += 1
                continue

            cold_size    = len(cold_set)
            similarities = {
                v: ov / (cold_size + len(entity_wishlist[v]) - ov)
                for v, ov in overlap_counts.items()
            }
            top = sorted(similarities, key=similarities.__getitem__, reverse=True)[:top_k]
            w   = torch.tensor([similarities[v] for v in top],
                               dtype=torch.float, device=device)
            w   = w / w.sum()
            idx = torch.tensor(top, dtype=torch.long, device=device)
            emb_matrix[eid] = (w.unsqueeze(1) * emb_matrix[idx]).sum(0)
            updated += 1

        return updated, skipped

    warm_users = set(range(model.num_users)) - cold_users
    warm_items = set(range(model.num_items)) - cold_items

    updated_u, skipped_u = jaccard_aggregate(
        list(cold_users), user_wishlist, item_wishlist,
        warm_users, model.user_embedding.weight.data
    )
    updated_i, skipped_i = jaccard_aggregate(
        list(cold_items), item_wishlist, user_wishlist,
        warm_items, model.item_embedding.weight.data
    )

    print(f"Users: {updated_u} initialized, {skipped_u} skipped")
    print(f"Items: {updated_i} initialized, {skipped_i} skipped")
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

        user_indices = edge_index[0]
        item_indices = edge_index[1]

        best_val = float("inf")
        best_weights = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            preds, u_init, i_init = self(user_indices, item_indices, edge_index, edge_index_t1, weights)
            task_loss = loss_fn(preds, targets)
            l2_reg = lambda_reg * (
                u_init.pow(2).sum(dim=1).mean() +
                i_init.pow(2).sum(dim=1).mean()
            )            
            loss = task_loss + l2_reg

            loss.backward()
            optimizer.step()

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

        if best_weights is not None:
            device = next(self.parameters()).device
            self.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
            print(f"Restored best weights | Monitor: {best_val:.4f}")
    @torch.no_grad()
    def print_attention_diagnostics(self, labels=None):
        """
        Prints the average attention weight per subgraph.
        'labels' should match the order of your subgraphs (e.g., ['R1', 'R2', 'R3', 'R4', 'R5', 'Wishlist'])
        """
        if self.attn_layer.last_weights is None:
            print("No attention weights recorded yet. Run a forward pass first.")
            return

        # weights shape: [N, K, 1] -> mean across nodes (dim 0)
        avg_weights = self.attn_layer.last_weights.mean(dim=0).squeeze().cpu().numpy()
        
        print("\n--- Attention Diagnostics (Last Batch) ---")
        if labels is None:
            labels = [f"Subgraph_{i}" for i in range(len(avg_weights))]
            
        for label, weight in zip(labels, avg_weights):
            print(f"{label:10} | Average Weight: {weight:.4f}")
        print("------------------------------------------\n")

    @torch.no_grad()
    def check_projection_diversity(self):
        """Measures how different the outputs of each projection layer are."""
        # Create a dummy user embedding [1, emb_dim]
        dummy_input = torch.randn(1, self.embedding_dim, device=self.user_embedding.weight.device)
        
        outputs = []
        for layer in self.rel_projections:
            outputs.append(layer(dummy_input).squeeze(0))
        
        # Calculate cosine similarity between all pairs
        # Sim(A, B) = (A dot B) / (||A|| * ||B||)
        print("\n--- Projection Divergence (Cosine Similarity) ---")
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                sim = F.cosine_similarity(outputs[i].unsqueeze(0), outputs[j].unsqueeze(0)).item()
                print(f"Layer {i} vs Layer {j}: {sim:.4f}")
        print("-------------------------------------------------\n")


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