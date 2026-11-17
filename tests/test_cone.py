import math

import numpy as np
import pytest
import torch

from utils.anchors import simplex_etf
from utils.cone import cone_radius, project_to_cone, remove_anchor_component, warmup_factor


def random_rows(classes=10, dim=128, scale=0.5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    anchors = simplex_etf(classes, dim, "cpu")
    weights = anchors + scale * torch.randn(classes, dim, generator=generator)
    bias = torch.randn(classes, generator=generator)
    return anchors, weights, bias


def test_removed_component_is_perpendicular_to_anchor():
    anchors, weights, _ = random_rows()
    for c in range(anchors.shape[0]):
        unit = anchors[c] / anchors[c].norm()
        perpendicular = remove_anchor_component(weights[c] - anchors[c], unit)
        assert abs(float(torch.dot(perpendicular, anchors[c]))) < 1e-5


@pytest.mark.parametrize("radius", [0.0, 0.05, 0.25, 0.9])
def test_projection_norm_angle_and_bias(radius):
    anchors, weights, bias = random_rows(scale=2.0)
    radii = torch.full((anchors.shape[0],), radius)
    project_to_cone(weights, anchors, radii, bias=bias)
    for c in range(anchors.shape[0]):
        a, w = anchors[c], weights[c]
        assert float(w.norm()) == pytest.approx(float(a.norm()), abs=1e-5)
        cosine = float(torch.dot(w, a) / (w.norm() * a.norm()))
        angle = math.acos(max(-1.0, min(1.0, cosine)))
        assert angle <= math.atan(radius) + 1e-3
        assert abs(float(bias[c])) <= radius * float(a.norm()) + 1e-7


def test_zero_radius_returns_anchor_and_zero_bias():
    anchors, weights, bias = random_rows()
    project_to_cone(weights, anchors, torch.zeros(anchors.shape[0]), bias=bias)
    assert torch.allclose(weights, anchors, atol=1e-6)
    assert torch.all(bias == 0)


def test_row_inside_cone_keeps_direction():
    anchors, _, _ = random_rows()
    weights = 1.7 * anchors.clone()
    project_to_cone(weights, anchors, torch.full((anchors.shape[0],), 0.25))
    assert torch.allclose(weights, anchors, atol=1e-6)


def test_radius_zero_for_missing_classes_and_bounded():
    counts = np.array([500, 0, 20, 300, 0, 5, 700, 0, 100, 50])
    radii, _ = cone_radius(counts, 10, max_radius=0.25, warmup=1.0)
    assert torch.all(radii[counts == 0] == 0)
    assert torch.all(radii <= 0.25 + 1e-7)
    assert torch.all(radii[counts > 0] > 0)
    # classes at or above the mean present count get the maximum radius
    mean_present = counts[counts > 0].mean()
    assert torch.allclose(radii[counts >= mean_present], torch.tensor(0.25))


def test_coverage_factor_halves_single_class_client():
    counts = np.zeros(10)
    counts[3] = 400
    radii, effective = cone_radius(counts, 10, max_radius=0.25, warmup=1.0)
    assert effective == pytest.approx(1.0)
    assert float(radii[3]) == pytest.approx(0.125)


def test_warmup_factor():
    assert warmup_factor(0, 20, 20) == 0.0
    assert warmup_factor(19, 20, 20) == 0.0
    assert warmup_factor(20, 20, 20) == pytest.approx(1 / 20)
    assert warmup_factor(39, 20, 20) == 1.0
    assert warmup_factor(150, 20, 20) == 1.0
