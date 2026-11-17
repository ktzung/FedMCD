"""Step 6 (Section 4.5): class-aware averaging (Eq. 19) of classifier rows and prototypes within a group."""

import torch
from torch.nn.utils import parameters_to_vector


def classifier_row(model, label):
    """Weight row and bias of class ``label`` (views into the model)."""
    return [model.fc.weight[label], model.fc.bias[label]]


def class_aware_average(server_model, anchors, label, group, presence):
    """Eq. 19: mean of the class-``label`` rows over G+ (the group members that hold the class).

    If G+ is empty, the row is a_c^(t+1) (bias 0); it is also written into the server
    classifier, from which the balanced classifiers start in the next round (Step 1).
    """
    holders = [client for client in group if label in presence.get(client.id, set())]
    server_row = classifier_row(server_model, label)
    if not holders:
        anchor = anchors[label]
        with torch.no_grad():
            for p in server_row:
                if p.shape == anchor.shape:
                    p.data.copy_(anchor)
                else:
                    p.data.zero_()
        return parameters_to_vector(server_row)

    averaged = torch.zeros_like(parameters_to_vector(server_row))
    for client in holders:
        averaged += parameters_to_vector(classifier_row(client.model, label))
    averaged /= len(holders)
    return averaged


def plain_average(server_model, label, group):
    """Mean over all group members, also those without the class (ablation without class-aware averaging)."""
    averaged = torch.zeros_like(parameters_to_vector(classifier_row(server_model, label)))
    for client in group:
        averaged += parameters_to_vector(classifier_row(client.model, label))
    averaged /= len(group)
    return averaged


def average_prototypes(label, group, presence):
    """Eq. 19 for prototypes: mean over G+, or None if G+ is empty (members keep their previous group prototype)."""
    holders = [client for client in group
               if label in presence.get(client.id, set()) and label in client.local_protos]
    if not holders:
        return None
    total = None
    for client in holders:
        if total is None:
            total = client.local_protos[label].clone()
        else:
            total = total + client.local_protos[label]
    return total / len(holders)


def set_classifier_row(model, label, vector):
    """Write a (weight row, bias) vector into class ``label`` of ``model``."""
    dim = model.fc.weight.shape[1]
    model.fc.weight.data[label] = vector[:dim].clone()
    model.fc.bias.data[label] = vector[dim:].view(()).clone()
