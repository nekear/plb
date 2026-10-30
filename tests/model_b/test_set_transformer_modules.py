import torch

from plb.model_b.ablations.set_transformer_modules import PMA, SAB

DIM = 16
HEADS = 4


def _tower() -> tuple[SAB, PMA]:
    torch.manual_seed(0)
    return SAB(DIM, DIM, HEADS, ln=True), PMA(DIM, HEADS, num_seeds=1, ln=True)


def test_pma_pooling_is_invariant_to_element_permutation() -> None:
    sab, pma = _tower()
    x = torch.randn(2, 7, DIM)
    permutation = torch.randperm(7)

    with torch.no_grad():
        pooled = pma(sab(x))
        pooled_permuted = pma(sab(x[:, permutation, :]))

    assert torch.allclose(pooled, pooled_permuted, atol=1e-5)


def test_sab_is_equivariant_to_element_permutation() -> None:
    sab, _ = _tower()
    x = torch.randn(2, 7, DIM)
    permutation = torch.randperm(7)

    with torch.no_grad():
        encoded = sab(x)
        encoded_permuted = sab(x[:, permutation, :])

    assert torch.allclose(encoded[:, permutation, :], encoded_permuted, atol=1e-5)


def test_padded_slots_do_not_change_the_pooled_output() -> None:
    sab, pma = _tower()
    real = torch.randn(2, 5, DIM)
    padded = torch.cat([real, torch.zeros(2, 3, DIM)], dim=1)
    mask = torch.tensor([[False] * 5 + [True] * 3] * 2)

    with torch.no_grad():
        pooled_real = pma(sab(real))
        pooled_padded = pma(sab(padded, mask), mask)

    assert torch.allclose(pooled_real, pooled_padded, atol=1e-5)


def test_an_all_padded_batch_row_produces_finite_output() -> None:
    sab, pma = _tower()
    x = torch.zeros(1, 4, DIM)
    mask = torch.ones(1, 4, dtype=torch.bool)  # every slot padded

    pooled = pma(sab(x, mask), mask)
    pooled.sum().backward()

    assert torch.isfinite(pooled).all()
    assert all(
        torch.isfinite(p.grad).all() for p in sab.parameters() if p.grad is not None
    )
