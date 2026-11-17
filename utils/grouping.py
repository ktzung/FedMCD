"""Step 5 (Section 4.5): residuals (Eq. 17), MADD between clients (Eq. 18) and DBSCAN, one class at a time."""

import numpy as np
from sklearn.cluster import DBSCAN


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(a.dot(b) / (na * nb))


def madd(vectors):
    """MADD of Eq. 18: mean absolute difference of cosine similarities to the other clients.

    D_ij = sum_{z != i, j} |cos(v_i, v_z) - cos(v_j, v_z)| / (n - 2); the cosine of a
    vector with norm below 1e-8 is 0. With fewer than three vectors, D_ij = 1 - cos(v_i, v_j).
    """
    vectors = np.asarray(vectors)
    num = len(vectors)
    if num < 3:
        result = np.zeros((num, num))
        for i in range(num):
            for j in range(i + 1, num):
                result[i][j] = result[j][i] = 1.0 - cosine(vectors[i], vectors[j])
        return result
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms >= 1e-8
    similarity = np.zeros((num, num))
    if valid.any():
        v = vectors[valid]
        nv = norms[valid]
        similarity[np.ix_(valid, valid)] = (v @ v.T) / np.outer(nv, nv)
    difference = np.abs(similarity[:, None, :] - similarity[None, :, :])
    index = np.arange(num)
    keep = (index[None, None, :] != index[:, None, None]) & (index[None, None, :] != index[None, :, None])
    result = np.where(keep, difference, 0.0).sum(axis=2) / (num - 2)
    np.fill_diagonal(result, 0.0)
    return result


def group_clients(classifier_weights, presence, anchors, eps, on_residuals=True):
    """Step 5: group the clients of every class by DBSCAN on MADD.

    ``classifier_weights`` maps client id to its balanced-classifier weight (C, d).
    With ``on_residuals`` the shared anchor is subtracted first and a missing
    class contributes a zero residual; otherwise a missing class contributes its
    anchor. Returns ``(groups, assignments)``: for each class the list of groups
    with at least two clients (sorted ids), and the DBSCAN label of every client.
    """
    anchors_np = anchors.cpu().numpy()
    if on_residuals:  # Eq. 17: residual w^bal_{k,c} - a_c^(t+1); zero vector for a missing class
        weights = {cid: (w.detach().clone() - anchors).cpu().numpy()
                   for cid, w in classifier_weights.items()}
    else:
        weights = {cid: w.detach().cpu().numpy() for cid, w in classifier_weights.items()}
    client_ids = list(classifier_weights.keys())
    num_classes, feature_dim = anchors_np.shape

    groups, assignments = {}, {}
    for c in range(num_classes):
        vectors = []
        for cid in client_ids:
            if c in presence.get(cid, set()):
                vectors.append(weights[cid][c])
            elif on_residuals:
                vectors.append(np.zeros(feature_dim))
            else:
                vectors.append(anchors_np[c])
        vectors = np.array(vectors)
        bad = ~np.isfinite(vectors).all(axis=1)
        if bad.any():
            vectors[bad] = anchors_np[c]

        distances = np.nan_to_num(madd(vectors), nan=0.0, posinf=1.0, neginf=0.0)
        labels = DBSCAN(eps=eps, min_samples=1, metric="precomputed").fit(distances).labels_
        assignments[c] = {cid: int(labels[i]) for i, cid in enumerate(client_ids)}
        merged = []
        for cluster in set(labels):
            members = np.where(labels == cluster)[0]
            if len(members) > 1:
                merged.append(sorted(client_ids[i] for i in members))
        groups[c] = merged
    return groups, assignments
