"""Initial simplex ETF anchors (Eq. 4) and the anchor update of Step 4 (Section 4.4).

Step 4 fills the class Gram matrix from the prototypes of classes held by at least
two clients (Eq. 13), blends it with the ETF Gram matrix (Eq. 14), projects it onto
a valid low-rank correlation matrix (Eq. 15), factors it into one vector per class,
rotates these vectors onto the prototypes (orthogonal Procrustes), and moves the
anchors towards them by an exponential moving average (Eq. 16).
"""

import torch

EPS = 1e-8


def simplex_etf(num_classes, feature_dim, device):
    """Initial anchors A^(1): unit-norm simplex ETF (Eq. 4), shape (C, d), with a random rotation."""
    c = num_classes
    if feature_dim < c - 1:
        raise ValueError(f"a simplex ETF for {c} classes needs feature_dim >= {c - 1}")
    device = torch.device(device)
    identity = torch.eye(c, device=device)
    ones = torch.ones(c, c, device=device)
    centered = torch.sqrt(torch.tensor(c / (c - 1), dtype=torch.float32, device=device)) * (
        identity - ones / c
    )
    left, _, _ = torch.linalg.svd(centered, full_matrices=False)
    rotation, _ = torch.linalg.qr(torch.randn(feature_dim, feature_dim, device=device))
    anchors = left[:, : c - 1] @ rotation[: c - 1]
    anchors = anchors / anchors.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return anchors.detach()


def normalize_rows(matrix):
    return matrix / matrix.norm(dim=1, keepdim=True).clamp_min(EPS)


def project_correlation(matrix, max_rank):
    """Pi_{r_G} of Eq. 15: symmetrize, keep the ``max_rank`` largest non-negative eigenvalues,
    reconstruct, set the diagonal to 1.

    Returns the projected matrix and its condition number over the retained subspace.
    """
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    rank = max(1, min(int(max_rank), matrix.shape[0]))
    kept = torch.argsort(eigenvalues, descending=True)[:rank]
    kept_values = eigenvalues[kept].clamp_min(0.0)
    kept_vectors = eigenvectors[:, kept]
    projected = (kept_vectors * kept_values.unsqueeze(0)) @ kept_vectors.T
    scale = projected.diag().clamp_min(EPS).sqrt()
    projected = projected / (scale[:, None] * scale[None, :])
    projected = 0.5 * (projected + projected.T)
    after = torch.linalg.eigvalsh(projected)
    floor = max(float(after.max().item()) * 1e-6, 1e-7)
    positive = after[after > floor]
    condition = float((positive.max() / positive.min()).item()) if positive.numel() > 1 else 1.0
    return projected, condition


def factor_and_rotate(gram, targets, trusted, feature_dim):
    """Factor the completed Gram matrix into class vectors and rotate them onto the prototypes (Procrustes)."""
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    rank = min(gram.shape[0] - 1, feature_dim)
    order = torch.argsort(eigenvalues, descending=True)[:rank]
    factors = eigenvectors[:, order] * eigenvalues[order].clamp_min(0.0).sqrt().unsqueeze(0)
    cross_covariance = factors[trusted].T @ targets[trusted]
    left, _, right_h = torch.linalg.svd(cross_covariance, full_matrices=False)
    return normalize_rows(factors @ (left @ right_h))


def update_anchors(initial_anchors, anchors, prototypes, supports, *, gram_weight, ema,
                   iterations, min_clients, max_residual, max_condition):
    """Step 4: one anchor update A^(t) -> A^(t+1). Returns ``(new_anchors, updated)``.

    ``prototypes[c]`` is the normalized mean prototype of class c and ``supports[c]``
    the number of participating clients that hold it. Classes with fewer than
    ``min_clients`` holders keep their anchors. The update is skipped when fewer
    than two classes qualify or the completed Gram matrix fails the residual or
    condition-number checks.
    """
    device, dtype = anchors.device, anchors.dtype
    base = normalize_rows(initial_anchors.to(device=device, dtype=dtype))
    previous = normalize_rows(anchors.to(device=device, dtype=dtype))
    observed = normalize_rows(prototypes.to(device=device, dtype=dtype))
    supports = supports.to(device=device)
    classes, feature_dim = previous.shape
    trusted = supports >= min_clients
    pair_mask = trusted[:, None] & trusted[None, :]
    observed_pairs = int(torch.triu(pair_mask, diagonal=1).sum().item())
    if trusted.sum().item() < 2 or observed_pairs == 0:
        return anchors, False
    if not torch.isfinite(observed).all():
        return anchors, False

    gram_etf = base @ base.T                    # G_ETF
    gram_observed = observed @ observed.T       # G_obs = Z Z^T
    observed_entries = pair_mask.to(dtype)      # R
    observed_entries.fill_diagonal_(1.0)

    completed = gram_etf.clone()
    condition = 1.0
    for _ in range(max(1, int(iterations))):
        mixed = observed_entries * gram_observed + (1.0 - observed_entries) * completed     # Eq. 13
        blended = (1.0 - gram_weight) * gram_etf + gram_weight * mixed                     # Eq. 14
        completed, condition = project_correlation(blended, max_rank=min(classes - 1, feature_dim))  # Eq. 15

    weighted_error = observed_entries * (completed - gram_observed)
    denominator = observed_entries.sum().clamp_min(1.0)
    residual = float(torch.sqrt(weighted_error.square().sum() / denominator).item())
    if not torch.isfinite(completed).all() or residual > max_residual or condition > max_condition:
        return anchors, False

    targets = previous.clone()
    targets[trusted] = observed[trusted]
    candidate = factor_and_rotate(completed, targets, trusted, feature_dim)    # A_cand
    candidate[~trusted] = previous[~trusted]
    updated = normalize_rows(ema * previous + (1.0 - ema) * candidate)         # Eq. 16
    if not torch.isfinite(updated).all():
        return anchors, False
    return updated.detach(), True


def mean_prototypes(clients, presence, anchors):
    """Z^(t) (normalized mean of the uploaded prototypes per class) and m_c (number of holders).

    Classes without a usable prototype keep their current anchor as placeholder.
    """
    classes = anchors.shape[0]
    device, dtype = anchors.device, anchors.dtype
    supports = torch.zeros(classes, device=device, dtype=torch.long)
    observed = anchors.clone()
    for label in range(classes):
        vectors = []
        for client in clients:
            if label in presence.get(client.id, set()) and label in client.local_protos:
                vector = client.local_protos[label].detach().to(device=device, dtype=dtype)
                if torch.isfinite(vector).all() and vector.norm().item() > 1e-8:
                    vectors.append(vector / vector.norm().clamp_min(1e-8))
        supports[label] = len(vectors)
        if vectors:
            mean = torch.stack(vectors).mean(dim=0)
            observed[label] = mean / mean.norm().clamp_min(1e-8)
    return observed, supports
