import torch
import torch.nn as nn

class GCNResidualEnsemble(nn.Module):
    """
    Two-phase training:
        phase 1 — fit_als():  train ALS alone, GCN untouched
        phase 2 — fit_gcn(): freeze ALS, train GCN + alpha

    Prediction:
        als_logit + alpha * gcn_logit

    The GCN is trained to predict the residual (target - ALS prediction),
    scaled by alpha so the two objectives are consistent.
    """

    def __init__(self, num_users, num_items, als_model, gcn_model):
        super().__init__()
        self.als   = als_model
        self.gcn   = gcn_model
        # BUG FIX 1: Initialize alpha to a small non-zero value.
        # alpha=0 causes zero gradient from main_loss to GCN params
        # (grad ∝ alpha), creating a dead cold-start where neither
        # GCN nor alpha can learn from each other.
        self.alpha = nn.Parameter(torch.tensor([0.1]))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, user_indices, item_indices, edge_index, weights):
        als_logit = self.als(user_indices, item_indices)
        gcn_logit, u_reg, i_reg = self.gcn(user_indices, item_indices, edge_index, weights)
        prediction = als_logit + self.alpha * gcn_logit
        return prediction, u_reg, i_reg

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_ratings(self, sids, pids, edge_index, weights):
        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
            s = torch.as_tensor(sids.copy(), dtype=torch.long, device=device)
            p = torch.as_tensor(pids.copy(), dtype=torch.long, device=device)
            out, _, _ = self.forward(s, p, edge_index, weights)
            return out.cpu().numpy()
        finally:
            self.train(was_training)

    # ------------------------------------------------------------------
    # Shared loop (used by fit_als)
    # ------------------------------------------------------------------

    def _fit(
        self,
        optimizer,
        forward_fn,
        targets,
        loss_fn,
        lambda_reg,
        epochs,
        val_fn,
        log_every,
        scheduler_patience,
        scheduler_factor,
        min_lr,
        early_stop_patience,
        min_delta,
        phase_label,
    ):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=scheduler_patience,
            factor=scheduler_factor, min_lr=min_lr,
        )

        best_val   = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            preds, l2 = forward_fn()
            loss = loss_fn(preds, targets) + lambda_reg * l2
            loss.backward()
            optimizer.step()

            monitor = val_fn() if val_fn is not None else loss.item()
            scheduler.step(monitor)

            if monitor < best_val - min_delta:
                best_val   = monitor
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch % log_every == 0:
                extra = f" | alpha: {self.alpha.item():.4f}" if phase_label == "GCN" else ""
                print(
                    f"[{phase_label}] Epoch {epoch:>5}"
                    f" | Loss: {loss.item():.4f}"
                    f" | Monitor: {monitor:.4f}"
                    f" | LR: {optimizer.param_groups[0]['lr']:.2e}"
                    f" | No improvement: {no_improve}/{early_stop_patience}"
                    + extra
                )

            if no_improve >= early_stop_patience:
                print(f"[{phase_label}] Early stopping at epoch {epoch} | Best: {best_val:.4f}")
                break

        if best_state is not None:
            device = next(self.parameters()).device
            self.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            print(f"[{phase_label}] Restored best weights | Monitor: {best_val:.4f}")

    # ------------------------------------------------------------------
    # Phase 1 — ALS only
    # ------------------------------------------------------------------

    def fit_als(
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
        device  = next(self.parameters()).device
        self.to(device)
        targets = targets.to(device)
        u_idx   = edge_index[0].to(device)
        i_idx   = edge_index[1].to(device)

        optimizer = torch.optim.Adam(self.als.parameters(), lr=lr)
        loss_fn   = loss_fn or nn.MSELoss()

        def forward_fn():
            preds = self.als(u_idx, i_idx)
            l2 = (
                self.als.user_emb.norm(2) ** 2 +
                self.als.item_emb.norm(2) ** 2
            )
            return preds, l2

        self._fit(
            optimizer=optimizer, forward_fn=forward_fn, targets=targets,
            loss_fn=loss_fn, lambda_reg=lambda_reg, epochs=epochs, val_fn=val_fn,
            log_every=log_every, scheduler_patience=scheduler_patience,
            scheduler_factor=scheduler_factor, min_lr=min_lr,
            early_stop_patience=early_stop_patience, min_delta=min_delta,
            phase_label="ALS",
        )

    # ------------------------------------------------------------------
    # Phase 2 — GCN residual (ALS frozen)
    # ------------------------------------------------------------------

    def fit_gcn(
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
        device     = next(self.parameters()).device
        self.to(device)
        edge_index = edge_index.to(device)
        weights    = weights.to(device)
        targets    = targets.to(device)
        u_idx      = edge_index[0]
        i_idx      = edge_index[1]

        # Freeze ALS parameters
        for param in self.als.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(
            list(self.gcn.parameters()) + [self.alpha], lr=lr
        )
        loss_fn = loss_fn or nn.MSELoss()

        # BUG FIX 3: Precompute the frozen ALS predictions and residual target
        # ONCE before the loop — they never change since ALS is frozen.
        # Recomputing them inside forward_fn() every epoch was wasteful and misleading.
        with torch.no_grad():
            als_logit       = self.als(u_idx, i_idx).detach()
            residual_target = (targets - als_logit).detach()

        def forward_fn():
            gcn_logit, u_reg, i_reg = self.gcn(u_idx, i_idx, edge_index, weights)
            preds = als_logit + self.alpha * gcn_logit
            l2    = u_reg.norm(2) ** 2 + i_reg.norm(2) ** 2

            # Main loss: full prediction vs targets (gradients flow to alpha AND GCN)
            main_loss = loss_fn(preds, targets)

            # BUG FIX 2: aux_loss must use (alpha * gcn_logit) vs residual_target,
            # NOT (gcn_logit) vs residual_target.
            # The prediction formula is als + alpha*gcn, so GCN should output
            # residual/alpha, not residual. Training gcn_logit → residual directly
            # conflicts with the alpha-scaled combination used in inference.
            # Using alpha here also keeps alpha in the aux gradient graph so
            # both parameters always receive a learning signal.
            aux_loss = loss_fn(self.alpha * gcn_logit, residual_target)

            return main_loss, aux_loss, l2

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=scheduler_patience,
            factor=scheduler_factor, min_lr=min_lr,
        )

        best_val   = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            main_loss, aux_loss, l2 = forward_fn()
            loss = main_loss + 0.5 * aux_loss + lambda_reg * l2
            loss.backward()
            optimizer.step()

            monitor = val_fn() if val_fn is not None else loss.item()
            scheduler.step(monitor)

            if monitor < best_val - min_delta:
                best_val   = monitor
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch % log_every == 0:
                print(
                    f"[GCN] Epoch {epoch:>5}"
                    f" | Loss: {loss.item():.4f}"
                    f" | Monitor: {monitor:.4f}"
                    f" | LR: {optimizer.param_groups[0]['lr']:.2e}"
                    f" | No improvement: {no_improve}/{early_stop_patience}"
                    f" | alpha: {self.alpha.item():.4f}"
                )

            if no_improve >= early_stop_patience:
                print(f"[GCN] Early stopping at epoch {epoch} | Best: {best_val:.4f}")
                break

        if best_state is not None:
            device = next(self.parameters()).device
            self.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            print(f"[GCN] Restored best weights | Monitor: {best_val:.4f}")