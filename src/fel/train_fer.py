"""Study 3: train, calibrate, export and evaluate a small image model on 48x48 faces.

Needs PyTorch (and timm for the pretrained backbone). Every function takes plain numpy
arrays of uint8 images with shape (n, 48, 48).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from fel.models import HAS_TORCH, require_torch

if HAS_TORCH:
    import torch
    import torch.nn.functional as F
    from torch import nn

    class FERWrapper(nn.Module):
        """Takes (B, 1, 48, 48) in 0..1. Normalises, upsamples to 96x96 and repeats to 3 channels
        inside the model, so the exported ONNX file needs only a resized grey crop as input."""

        def __init__(self, backbone: nn.Module, size: int = 96):
            super().__init__()
            self.backbone, self.size = backbone, size

        def forward(self, x):
            x = (x - 0.5) / 0.5
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear",
                              align_corners=False)
            return self.backbone(x.repeat(1, 3, 1, 1))


def build_model(name: str = "mobilenetv3_small_100", n_classes: int = 8,
                pretrained: bool = True):
    """Backbone from timm wrapped with the preprocessing above. Check the pretrained
    weights' licence on the timm model card before publishing anything derived from them."""
    require_torch()
    try:
        import timm
    except ImportError as exc:  # pragma: no cover
        raise ImportError("timm is required. Run: pip install -e '.[torch]'") from exc
    return FERWrapper(timm.create_model(name, pretrained=pretrained, num_classes=n_classes))


def _to_tensor(x_uint8: np.ndarray, dev):
    return torch.tensor(x_uint8, dtype=torch.float32, device=dev).div_(255.0).unsqueeze(1)


def _augment(x):
    """Random horizontal flip and a random shift of up to 4 pixels, on a (B, 1, 48, 48) batch."""
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    pad = F.pad(x, (4, 4, 4, 4), mode="replicate")
    dx, dy = np.random.randint(0, 9, size=2)
    return pad[:, :, dy : dy + 48, dx : dx + 48]


def predict_logits(model, x_uint8: np.ndarray, batch_size: int = 256, device: str | None = None):
    require_torch()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_uint8), batch_size):
            out.append(model(_to_tensor(x_uint8[i : i + batch_size], dev)).cpu().numpy())
    return np.concatenate(out)


def train_classifier(
    model, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
    n_classes: int = 8, epochs: int = 15, lr: float = 1e-3, batch_size: int = 128,
    label_smoothing: float = 0.1, balance: bool = True, augment: bool = True, seed: int = 0,
    device: str | None = None,
) -> dict[str, Any]:
    """Train and keep the checkpoint with the best validation macro F1 (reloaded at the end).

    Selection uses the validation split only. The test split is never seen here.
    """
    require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = None
    if balance:
        counts = np.bincount(y_train, minlength=n_classes)
        w = torch.tensor(len(y_train) / (n_classes * np.maximum(counts, 1)),
                         dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
    yt = torch.tensor(y_train, dtype=torch.long, device=dev)
    best_f1, best_state, best_epoch, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_train), device=dev)
        for i in range(0, len(perm), batch_size):
            b = perm[i : i + batch_size]
            if len(b) < 2:
                continue
            xb = _to_tensor(x_train[b.cpu().numpy()], dev)
            if augment:
                xb = _augment(xb)
            opt.zero_grad()
            crit(model(xb), yt[b]).backward()
            opt.step()
        sched.step()
        logits = predict_logits(model, x_val, device=str(dev))
        f1 = float(f1_score(y_val, logits.argmax(1), labels=list(range(n_classes)),
                            average="macro", zero_division=0))
        history.append({"epoch": epoch, "val_macro_f1": f1})
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_val_macro_f1": best_f1, "history": history}


def export_onnx(model, path: str | Path, opset: int = 17) -> Path:
    """Export with a dynamic batch axis. Input: (B, 1, 48, 48) float32 in 0..1."""
    require_torch()
    model = model.cpu().eval()
    dummy = torch.zeros(1, 1, 48, 48)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=opset,
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:  # newer PyTorch versions default to a different exporter
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, dummy, str(path), **kw)
    return path


def quantize_int8(src: str | Path, dst: str | Path) -> Path | None:
    """Dynamic INT8 weight quantisation. Returns None if onnxruntime cannot do it for this model.

    Check the quantised model's accuracy and latency yourself. It may be no faster.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
        return Path(dst)
    except Exception as exc:  # noqa: BLE001
        print(f"INT8 quantisation failed: {exc}")
        return None


def onnx_session(path: str | Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def onnx_logits(session, x_uint8: np.ndarray) -> np.ndarray:
    x = (x_uint8.astype(np.float32) / 255.0)[:, None, :, :]
    return session.run(["logits"], {"input": x})[0]
