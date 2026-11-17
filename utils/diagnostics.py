"""Per-round diagnostics of FedMCD (Additional file 1, Section S5). They do not affect training."""

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score

from utils.grouping import madd


def client_residual_record(client, radii):
    """Residual statistics of one client's local classifier after Step 2."""
    with torch.no_grad():
        anchors = client.anchors
        weight = client.model.fc.weight
        ratios = (weight - anchors).norm(dim=1) / anchors.norm(dim=1).clamp_min(1e-12)
        present = torch.zeros(client.cfg.num_classes, dtype=torch.bool, device=weight.device)
        if client.present_classes:
            present[list(client.present_classes)] = True
        missing = ~present
        eligible = present & (radii > 1e-12)
        return {
            "present_classes": len(client.present_classes),
            "present_residual": float(ratios[present].mean().item()) if present.any() else 0.0,
            "missing_residual_max": float(ratios[missing].max().item()) if missing.any() else 0.0,
            "boundary_fraction": (float((ratios[eligible] >= 0.95 * radii[eligible]).float().mean().item())
                                  if eligible.any() else 0.0),
        }


class Diagnostics:
    """Collects one row per round: residual geometry, MADD and temporal NMI of the groups."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.rows = []
        self.client_records = []
        self.previous_assignments = {}

    def record_client(self, client, radii):
        self.client_records.append(client_residual_record(client, radii))

    def end_round(self, round_index, assignments, presence, balanced_classifiers, anchors):
        records, self.client_records = self.client_records, []
        presence_rates, temporal_nmis, madd_means = [], [], []
        for label in range(self.num_classes):
            labels_by_client = assignments.get(label, {})
            if not labels_by_client:
                continue
            ids = sorted(labels_by_client)
            presence_rates.append(sum(label in presence.get(cid, set()) for cid in ids) / max(1, len(ids)))
            previous = self.previous_assignments.get(label, {})
            common = sorted(set(ids) & set(previous))
            if len(common) >= 2:
                temporal_nmis.append(normalized_mutual_info_score(
                    [previous[cid] for cid in common], [labels_by_client[cid] for cid in common],
                    average_method="arithmetic"))
            vectors = []
            for cid in ids:
                if label in presence.get(cid, set()):
                    residual = balanced_classifiers[cid][0][label].detach().to(anchors.device) - anchors[label]
                    vectors.append(residual.cpu().numpy())
                else:
                    vectors.append(np.zeros(anchors.shape[1], dtype=np.float32))
            if len(vectors) >= 2:
                distances = madd(np.asarray(vectors))
                upper = distances[np.triu_indices(len(vectors), k=1)]
                if upper.size:
                    madd_means.append(float(np.mean(upper)))
        self.previous_assignments = {label: dict(a) for label, a in assignments.items()}

        def mean(values):
            return float(sum(values) / len(values)) if values else 0.0

        self.rows.append({
            "round": int(round_index),
            "mean_present_classes": mean([r["present_classes"] for r in records]),
            "class_presence_rate": mean(presence_rates),
            "mean_present_residual": mean([r["present_residual"] for r in records]),
            "max_missing_residual": max((r["missing_residual_max"] for r in records), default=0.0),
            "mean_boundary_fraction": mean([r["boundary_fraction"] for r in records]),
            "mean_madd": mean(madd_means),
            "temporal_nmi": mean(temporal_nmis),
        })
