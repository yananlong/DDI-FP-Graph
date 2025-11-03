import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from GPU.models import FPMLP


def _make_binary(batch: int, dim: int) -> torch.Tensor:
    return torch.randint(0, 2, (batch, dim), dtype=torch.float32)


def test_fingerprint_symmetric_is_order_invariant():
    model = FPMLP(
        in_dim=16,
        hid_dim=8,
        out_dim=3,
        nlayers=1,
        dropout=0.0,
        fusion="fingerprint_symmetric",
    )
    model.eval()

    drug_a = _make_binary(batch=4, dim=16)
    drug_b = _make_binary(batch=4, dim=16)

    with torch.no_grad():
        logits_ab = model(drug_a, drug_b)
        logits_ba = model(drug_b, drug_a)

    assert torch.allclose(logits_ab, logits_ba, atol=1e-6)


def test_embedding_symmetric_is_order_invariant():
    model = FPMLP(
        in_dim=16,
        hid_dim=8,
        out_dim=3,
        nlayers=1,
        dropout=0.0,
        fusion="embedding_symmetric",
    )
    model.eval()

    drug_a = _make_binary(batch=4, dim=16)
    drug_b = _make_binary(batch=4, dim=16)

    with torch.no_grad():
        logits_ab = model(drug_a, drug_b)
        logits_ba = model(drug_b, drug_a)

    assert torch.allclose(logits_ab, logits_ba, atol=1e-6)


def test_invalid_fusion_raises_value_error():
    with pytest.raises(ValueError):
        FPMLP(
            in_dim=16,
            hid_dim=8,
            out_dim=3,
            nlayers=1,
            dropout=0.0,
            fusion="invalid",
        )
