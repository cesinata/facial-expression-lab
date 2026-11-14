"""Run only where PyTorch is installed (CI torch job, Colab, your machine)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fel.models import FERNet, TorchFERClassifier  # noqa: E402


def test_fernet_output_shape_for_odd_and_even_kernels():
    for k in (2, 3, 4, 5, 6):
        net = FERNet(30, 5, hidden=24, channels=4, kernel=k).eval()
        assert net(torch.zeros(7, 30)).shape == (7, 5)


def _toy(n=300, f=10, k=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    X = rng.normal(size=(n, f)) + y[:, None] * 1.5
    return X, y


def test_classifier_learns_and_reloads_best_checkpoint():
    X, y = _toy()
    clf = TorchFERClassifier(3, hidden=16, channels=4, max_epochs=40, patience=5, seed=0)
    clf.fit(X[:200], y[:200], X[200:250], y[200:250])
    assert 1 <= clf.best_epoch_ <= clf.stopped_epoch_ <= 40
    proba = clf.predict_proba(X[250:])
    assert proba.shape == (50, 3) and np.allclose(proba.sum(1), 1, atol=1e-5)
    assert (proba.argmax(1) == y[250:]).mean() > 0.7


def test_classifier_needs_validation_data():
    X, y = _toy()
    with pytest.raises(ValueError):
        TorchFERClassifier(3).fit(X, y)


def test_train_classifier_and_onnx_export_on_a_tiny_image_model(tmp_path):
    from torch import nn

    from fel import train_fer as T

    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 120)
    x = (rng.random((120, 48, 48)) * 60 + y[:, None, None] * 60).astype(np.uint8)
    model = nn.Sequential(nn.Flatten(), nn.Linear(48 * 48, 3))
    out = T.train_classifier(model, x[:80], y[:80], x[80:], y[80:], n_classes=3, epochs=3,
                             batch_size=32, augment=False, device="cpu")
    assert 1 <= out["best_epoch"] <= 3 and len(out["history"]) == 3
    logits = T.predict_logits(model, x[80:], device="cpu")
    assert logits.shape == (40, 3)
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    path = T.export_onnx(model, tmp_path / "m.onnx")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = T.onnx_logits(sess, x[80:])
    assert np.allclose(got, logits, atol=1e-3)
