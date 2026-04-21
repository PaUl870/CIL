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
        return torch.sigmoid(self.net(torch.cat([u, i], dim=-1)).squeeze(-1))


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
        mlp_hidden_dims: list[int] = (256, 64), 
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.attn_layer = RatingAttention(embedding_dim)

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

        # Prediction head
        self.mlp_head = (
            PredictionMLP(embedding_dim, list(mlp_hidden_dims))
            if use_mlp_head
            else None
        )

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def _score(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """Unified scoring: MLP head or scaled dot product."""
        if self.mlp_head is not None:
            return self.mlp_head(u, i)
        return torch.sigmoid((u * i).sum(dim=-1) / math.sqrt(self.embedding_dim))

    def forward(self, user_indices, item_indices, edge_index, weights):
        subgraphs = [
            {"edge_index": edge_index[:, weights == r]}
            for r in torch.unique(weights)
        ]

        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight

        all_u = [u_emb]
        all_i = [i_emb]

        for layer in self.layers:
            sub_msgs_u, sub_msgs_i = [], []
            for graph in subgraphs:
                w_dummy = torch.ones(graph["edge_index"].size(1), device=edge_index.device)
                u_msg, i_msg = layer(u_emb, i_emb, graph["edge_index"], w_dummy)
                sub_msgs_u.append(u_msg)
                sub_msgs_i.append(i_msg)

            u_emb = self.attn_layer(u_emb, sub_msgs_u)
            i_emb = self.attn_layer(i_emb, sub_msgs_i)

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
    def predict_ratings(self, sids, pids, edge_index, weights):
        """Predict ratings for specific user-item pairs."""
        self.eval()
        device = next(self.parameters()).device

        s_tensor = torch.as_tensor(sids.copy(), dtype=torch.long, device=device)
        p_tensor = torch.as_tensor(pids.copy(), dtype=torch.long, device=device)

        out, _, _ = self.forward(s_tensor, p_tensor, edge_index, weights)
        return out.cpu().numpy()

    def fit(
        self,
        edge_index,
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


class RatingAttention(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.attn_project = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1),
        )

    def forward(self, query_emb, subgraph_messages):
        """
        Args:
            query_emb:          [N, dim]
            subgraph_messages:  list of K tensors, each [N, dim]
        Returns:
            [N, dim] attention-weighted sum of messages
        """
        stacked = torch.stack(subgraph_messages, dim=1)   # [N, K, dim]
        K = stacked.size(1)

        query_expanded = query_emb.unsqueeze(1).expand(-1, K, -1)  # [N, K, dim]
        scores = self.attn_project(torch.cat([query_expanded, stacked], dim=-1))  # [N, K, 1]
        weights = F.softmax(scores, dim=1)

        return (stacked * weights).sum(dim=1)  # [N, dim]