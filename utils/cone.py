"""Cone radius (Pre-step, Section 4.2) and cone projection (Eqs. 9-11, Section 4.3)."""

import torch


def warmup_factor(round_index, warmup_rounds, ramp_rounds):
    """Warm-up factor xi_t: 0 during the first T_warm rounds, then linear to 1 over T_inc rounds."""
    if round_index < warmup_rounds:
        return 0.0
    if ramp_rounds <= 0:
        return 1.0
    return min(1.0, max(0.0, (round_index - warmup_rounds + 1) / ramp_rounds))


def cone_radius(label_counts, num_classes, *, max_radius, kappa=1.0, coverage_classes=2.0,
                warmup=1.0, device=None, dtype=torch.float32):
    """Cone radius rho_{k,c} of every class (Eq. 7).

    rho_c = rho_max * coverage factor (Eq. 6) * warm-up factor xi_t * frequency factor_c (Eq. 5),
    and 0 for classes without local samples. Returns ``(radii, effective_classes)``.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if max_radius < 0 or kappa <= 0 or coverage_classes <= 0:
        raise ValueError("cone radius parameters must be positive")
    counts = torch.as_tensor(label_counts, device=device, dtype=dtype).flatten()
    if counts.numel() != num_classes:
        raise ValueError(f"expected {num_classes} class counts, got {counts.numel()}")
    if (counts < 0).any():
        raise ValueError("label counts must be non-negative")
    total = counts.sum()
    if total.item() <= 0:
        return torch.zeros_like(counts), 0.0

    present = counts > 0
    mean_present = counts[present].mean()
    relative = torch.zeros_like(counts)
    relative[present] = counts[present] / mean_present
    frequency_factor = ((1.0 + kappa) * relative / (relative + kappa)).clamp(0.0, 1.0)  # Eq. 5, chi = relative

    probabilities = counts[present] / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    effective_classes = float(torch.exp(entropy).item())
    coverage_factor = min(1.0, effective_classes / float(coverage_classes))  # Eq. 6
    warmup = max(0.0, min(1.0, float(warmup)))
    radii = float(max_radius) * coverage_factor * warmup * frequency_factor  # Eq. 7
    return radii, effective_classes


def remove_anchor_component(residual, anchor_unit):
    """v_perp = v - (v^T a_hat) a_hat: the residual without its component along the anchor (Eq. 9)."""
    return residual - torch.dot(residual, anchor_unit) * anchor_unit


def project_to_cone(weights, anchors, radii, bias=None, epsilon=1e-12):
    """Cone projection Pi_B(a_c, rho_{k,c}) of every classifier row, in place (Eqs. 9-11).

    Eq. 9:  the residual v = w - a loses its component along the anchor;
    Eq. 10: the perpendicular part is shortened to at most rho * |a|, giving u;
    Eq. 11: a + u is rescaled to the anchor norm.
    The bias is clipped to [-rho * |a|, rho * |a|]. A zero radius returns the anchor and a zero bias.
    """
    if weights.shape != anchors.shape or weights.ndim != 2:
        raise ValueError("weights and anchors must be aligned matrices")
    if radii.ndim != 1 or radii.numel() != weights.shape[0]:
        raise ValueError("radii must contain one value per class")
    if not torch.isfinite(anchors).all():
        raise ValueError("anchors must be finite")
    with torch.no_grad():
        for label in range(weights.shape[0]):
            anchor = anchors[label]
            anchor_norm = anchor.norm().clamp_min(epsilon)
            anchor_unit = anchor / anchor_norm
            radius = max(0.0, float(radii[label].item()))
            maximum_norm = radius * anchor_norm
            residual = weights[label] - anchor                                  # v
            perpendicular = remove_anchor_component(residual, anchor_unit)     # Eq. 9
            perpendicular_norm = perpendicular.norm()
            if perpendicular_norm > maximum_norm:                              # Eq. 10
                perpendicular.mul_(maximum_norm / perpendicular_norm.clamp_min(epsilon))
            candidate = anchor + perpendicular                                 # Eq. 11
            weights[label].copy_(candidate * (anchor_norm / candidate.norm().clamp_min(epsilon)))
            if bias is not None:
                bias[label].clamp_(-maximum_norm, maximum_norm)
