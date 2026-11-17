import numpy as np
import pytest
import torch

from utils.anchors import simplex_etf, update_anchors
from utils.grouping import group_clients, madd


def test_simplex_etf_geometry():
    torch.manual_seed(0)
    anchors = simplex_etf(10, 128, "cpu")
    gram = anchors @ anchors.T
    assert torch.allclose(gram.diag(), torch.ones(10), atol=1e-5)
    off = gram[~torch.eye(10, dtype=torch.bool)]
    assert torch.allclose(off, torch.full_like(off, -1 / 9), atol=1e-5)


def test_anchor_update_keeps_unit_norm_and_skips_without_support():
    torch.manual_seed(0)
    anchors = simplex_etf(10, 32, "cpu")
    prototypes = torch.nn.functional.normalize(anchors + 0.3 * torch.randn(10, 32), dim=1)
    kwargs = dict(gram_weight=0.25, ema=0.9, iterations=5, min_clients=2,
                  max_residual=1.0, max_condition=1e6)
    new, updated = update_anchors(anchors, anchors, prototypes, torch.full((10,), 5), **kwargs)
    assert updated
    assert torch.allclose(new.norm(dim=1), torch.ones(10), atol=1e-5)
    supports = torch.zeros(10, dtype=torch.long)
    supports[0] = 5
    same, updated = update_anchors(anchors, anchors, prototypes, supports, **kwargs)
    assert not updated and same is anchors


def madd_reference(vectors):
    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return 0.0 if na < 1e-8 or nb < 1e-8 else float(a.dot(b) / (na * nb))
    n = len(vectors)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = sum(abs(cos(vectors[i], vectors[z]) - cos(vectors[j], vectors[z]))
                    for z in range(n) if z not in (i, j))
            out[i, j] = out[j, i] = d / (n - 2)
    return out


def test_madd_matches_definition():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(12, 16))
    vectors[3] = 0.0
    assert np.allclose(madd(vectors), madd_reference(vectors), atol=1e-12)


def test_grouping_separates_two_concepts_and_ignores_missing_classes():
    torch.manual_seed(0)
    anchors = simplex_etf(3, 16, "cpu")
    direction = torch.zeros(16)
    direction[0] = 1.0
    weights, presence = {}, {}
    for cid in range(8):
        w = anchors.clone()
        w[1] += (0.2 if cid < 4 else -0.2) * direction
        weights[cid] = w
        presence[cid] = {0, 1, 2} if cid != 7 else {0, 2}
    groups, assignments = group_clients(weights, presence, anchors, eps=0.1, on_residuals=True)
    first, second = assignments[1][0], assignments[1][4]
    assert first != second
    assert all(assignments[1][cid] == first for cid in range(4))
    assert all(assignments[1][cid] == second for cid in (4, 5, 6))
    assert all(len(group) >= 2 for group in groups[1])
