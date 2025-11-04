import pytest


np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from GBDT.models import BaseFingerprintGBDT


class _DummyEstimator:
    def __init__(self, out_dim: int) -> None:
        self.out_dim = out_dim
        self.fitted = False

    def fit(self, features, labels):
        self.fitted = True
        return self

    def predict_proba(self, features):
        features = np.asarray(features)
        if features.ndim == 1:
            features = features[np.newaxis, :]
        probs = np.ones((features.shape[0], self.out_dim), dtype=np.float32)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs


class DummyGBDT(BaseFingerprintGBDT):
    def __init__(self, fusion: str) -> None:
        super().__init__(out_dim=3, fusion=fusion)

    def _build_estimator(self, out_dim: int):
        return _DummyEstimator(out_dim)


def _make_binary(batch: int, dim: int) -> torch.Tensor:
    return torch.randint(0, 2, (batch, dim), dtype=torch.float32)


def test_fingerprint_symmetric_is_order_invariant():
    model = DummyGBDT(fusion="fingerprint_symmetric")

    drug_a = _make_binary(batch=4, dim=8)
    drug_b = _make_binary(batch=4, dim=8)

    fused_ab = model._fuse_inputs(drug_a, drug_b)
    fused_ba = model._fuse_inputs(drug_b, drug_a)

    assert torch.allclose(fused_ab, fused_ba, atol=1e-6)


def test_fingerprint_concat_preserves_order():
    model = DummyGBDT(fusion="fingerprint_concat")

    drug_a = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    drug_b = torch.tensor([[0.0, 1.0, 0.0, 1.0]], dtype=torch.float32)

    fused_ab = model._fuse_inputs(drug_a, drug_b)
    fused_ba = model._fuse_inputs(drug_b, drug_a)

    assert fused_ab.shape[-1] == drug_a.shape[-1] * 2
    assert not torch.allclose(fused_ab, fused_ba)


def test_prepare_features_returns_numpy_float32():
    model = DummyGBDT(fusion="fingerprint_symmetric")

    drug_a = torch.ones((2, 6), dtype=torch.float32)
    drug_b = torch.zeros((2, 6), dtype=torch.float32)

    features = model._prepare_features(drug_a, drug_b)

    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float32
    assert features.shape == (2, 18)


def test_invalid_fusion_raises_value_error():
    with pytest.raises(ValueError):
        DummyGBDT(fusion="invalid")
