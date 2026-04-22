import torch
import torch.nn as nn

class ALS(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        lambda_reg: float = 1e-1,
    ):
        super().__init__()
        self.k = embedding_dim
        self.lambda_reg = lambda_reg

        # Initialization: Using 1/sqrt(k) prevents initial "exploding" dot products
        scale = 1.0 / (embedding_dim ** 0.5)
        self.user_emb = nn.Parameter(torch.randn(num_users, embedding_dim) * scale)
        self.item_emb = nn.Parameter(torch.randn(num_items, embedding_dim) * scale)

        self.user_bias = nn.Parameter(torch.zeros(num_users))
        self.item_bias = nn.Parameter(torch.zeros(num_items))
        self.register_buffer("global_mean", torch.tensor(0.0))

    def forward(self, user_indices, item_indices):
        u = self.user_emb[user_indices]
        i = self.item_emb[item_indices]
        dot = (u * i).sum(dim=-1)
        return self.global_mean + self.user_bias[user_indices] + self.item_bias[item_indices] + dot

    @torch.no_grad()
    def predict_ratings(self, sids, pids, *args, **kwargs):
        self.eval()
        device = self.user_emb.device
        s = torch.as_tensor(sids, dtype=torch.long, device=device)
        p = torch.as_tensor(pids, dtype=torch.long, device=device)
        return self.forward(s, p).cpu().numpy()

    @torch.no_grad()
    def fit_als(self, edge_index, weights, epochs=20, val_fn=None, log_every=1):
        device = self.user_emb.device
        u_idx, i_idx = edge_index[0].to(device), edge_index[1].to(device)
        ratings = weights.to(device).float()

        self.global_mean.fill_(ratings.mean())
        num_users, num_items = self.user_emb.shape[0], self.item_emb.shape[0]

        for epoch in range(epochs):
            # Update Users (Bias + Factors)
            targets_u = ratings - self.global_mean - self.item_bias[i_idx]
            self.user_bias.data, self.user_emb.data = self._solve_step(
                known_emb=self.item_emb,
                known_idx=i_idx,
                solve_idx=u_idx,
                targets=targets_u,
                num_solve=num_users
            )

            # Update Items (Bias + Factors)
            targets_i = ratings - self.global_mean - self.user_bias[u_idx]
            self.item_bias.data, self.item_emb.data = self._solve_step(
                known_emb=self.user_emb,
                known_idx=u_idx,
                solve_idx=i_idx,
                targets=targets_i,
                num_solve=num_items
            )

            if epoch % log_every == 0 and val_fn:
                print(f"Epoch {epoch} | Monitor: {val_fn():.4f}")

    @torch.no_grad()
    def _solve_step(self, known_emb, known_idx, solve_idx, targets, num_solve):
        device = known_emb.device
        k = self.k
        
        # Build V (the "features" matrix for the ridge solve)
        ones = torch.ones(known_emb.shape[0], 1, device=device)
        V_all = torch.cat([ones, known_emb], dim=1) 
        V = V_all[known_idx] 

        # 1. Accumulate the Gram Matrix (A) and Right-Hand Side (rhs)
        A = torch.zeros(num_solve, k + 1, k + 1, device=device)
        rhs = torch.zeros(num_solve, k + 1, device=device)

        # Vectorized outer product: (N, k+1, 1) * (N, 1, k+1) -> (N, k+1, k+1)
        VtV = V.unsqueeze(2) * V.unsqueeze(1)
        A.scatter_add_(0, solve_idx.view(-1, 1, 1).expand(-1, k+1, k+1), VtV)
        rhs.scatter_add_(0, solve_idx.unsqueeze(1).expand(-1, k + 1), V * targets.unsqueeze(1))

        # 2. Precise Regularization
        # We need a diagonal matrix where reg[0] is for the bias and reg[1:] for factors
        counts = torch.zeros(num_solve, device=device)
        counts.scatter_add_(0, solve_idx, torch.ones_like(solve_idx, dtype=torch.float))
        
        # This is the "Secret Sauce":
        # Bias needs VERY LITTLE reg (1e-4) to allow it to absorb the mean residual
        # Factors need the FULL reg (lambda * counts)
        reg_weights = torch.full((k + 1,), self.lambda_reg, device=device)
        reg_weights[0] = 1e-4 
        
        # Apply the diagonal regularization
        # total_reg = lambda * count * reg_weights
        total_reg = counts.view(-1, 1) * reg_weights.view(1, -1)
        
        # Add a tiny floor (1e-6) to the whole diagonal for absolute numerical stability
        A.diagonal(dim1=1, dim2=2).add_(total_reg + 1e-6)

        # 3. Solve the system Ax = b
        # Using linalg.solve is faster than inv() and more accurate than try/except fallback
        sol = torch.linalg.solve(A, rhs)
        
        return sol[:, 0], sol[:, 1:]