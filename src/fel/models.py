"""A small 1D CNN for tabular landmark features (PyTorch), with the early-stopping fix.

This is a re-implementation of the thesis block structure (linear block, then a
convolution block over the hidden vector, then a classifier). It is not the thesis
model or its weights. The difference that matters: the best-validation checkpoint is
reloaded before prediction. The thesis code saved that checkpoint but evaluated the
last-epoch weights.

PyTorch is imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover - depends on the environment
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False


def require_torch() -> None:
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for this model. Run: pip install -e '.[torch]'")


if HAS_TORCH:

    class FERNet(nn.Module):
        def __init__(self, n_features: int, n_classes: int, hidden: int = 64, channels: int = 8,
                     kernel: int = 3, dropout: float = 0.3):
            super().__init__()
            self.linear = nn.Sequential(
                nn.Linear(n_features, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
                nn.Dropout(dropout),
            )
            pad = kernel // 2
            self.conv = nn.Sequential(
                nn.Conv1d(1, channels, kernel, padding=pad), nn.ReLU(),
                nn.BatchNorm1d(channels), nn.MaxPool1d(2), nn.Dropout(dropout),
            )
            conv_len = hidden + 2 * pad - kernel + 1
            self.head = nn.Linear(channels * (conv_len // 2), n_classes)

        def forward(self, x):
            h = self.linear(x).unsqueeze(1)
            return self.head(self.conv(h).flatten(1))


class TorchFERClassifier:
    """fit(X, y, X_val, y_val) with early stopping on validation loss; predict_proba(X)."""

    def __init__(self, n_classes: int, hidden: int = 64, channels: int = 8, kernel: int = 3,
                 dropout: float = 0.3, lr: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 64, max_epochs: int = 100, patience: int = 10,
                 balance: bool = False, seed: int = 0, device: str | None = None):
        self.n_classes, self.hidden, self.channels = n_classes, hidden, channels
        self.kernel, self.dropout, self.lr = kernel, dropout, lr
        self.weight_decay, self.batch_size = weight_decay, batch_size
        self.max_epochs, self.patience, self.balance = max_epochs, patience, balance
        self.seed, self.device = seed, device

    def fit(self, X, y, X_val=None, y_val=None):
        require_torch()
        if X_val is None or y_val is None:
            raise ValueError("a validation set is required for early stopping")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        model = FERNet(X.shape[1], self.n_classes, self.hidden, self.channels, self.kernel,
                       self.dropout).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        w = None
        if self.balance:
            counts = np.bincount(y, minlength=self.n_classes)
            w = torch.tensor(len(y) / (self.n_classes * np.maximum(counts, 1)),
                             dtype=torch.float32, device=dev)
        crit = torch.nn.CrossEntropyLoss(weight=w)
        Xt, yt = torch.tensor(X, device=dev), torch.tensor(y, device=dev)
        Xv = torch.tensor(np.asarray(X_val, dtype=np.float32), device=dev)
        yv = torch.tensor(np.asarray(y_val, dtype=np.int64), device=dev)
        best, best_state, bad = float("inf"), None, 0
        self.history_: list[dict[str, float]] = []
        for epoch in range(1, self.max_epochs + 1):
            model.train()
            perm = torch.randperm(len(Xt), device=dev)
            for i in range(0, len(perm), self.batch_size):
                b = perm[i : i + self.batch_size]
                if len(b) < 2:  # BatchNorm needs at least two rows
                    continue
                opt.zero_grad()
                crit(model(Xt[b]), yt[b]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(crit(model(Xv), yv))
            self.history_.append({"epoch": epoch, "val_loss": val_loss})
            if val_loss < best - 1e-6:
                best, bad, self.best_epoch_ = val_loss, 0, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.stopped_epoch_ = epoch
        model.load_state_dict(best_state)  # reload the best checkpoint before prediction
        self.model_ = model.eval()
        self._dev = dev
        return self

    def predict_proba(self, X) -> np.ndarray:
        require_torch()
        X = torch.tensor(np.asarray(X, dtype=np.float32), device=self._dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 1024):
                out.append(torch.softmax(self.model_(X[i : i + 1024]), dim=1).cpu().numpy())
        return np.concatenate(out)
