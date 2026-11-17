"""FedDrift (Jothimurugesan et al., AISTATS 2023): drift detection, new clusters and cluster merging."""

import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from entities.IFCA import IFCAClient, IFCAServer
from utils.metrics import average_loss


class FedDriftClient(IFCAClient):
    def __init__(self, client_id, cfg, train_set, test_set):
        super().__init__(client_id, cfg, train_set, test_set)
        self.prev_train_set = train_set

    def detect_and_choose(self, global_models):
        """Cluster with the lowest loss, or None (new cluster) if the best loss rose above the threshold."""
        prev_losses = [average_loss(m, self.prev_train_set, self.criterion, self.device) for m in global_models]
        min_prev_loss = min(prev_losses)
        losses = [average_loss(m, self.train_set, self.criterion, self.device) for m in global_models]
        if min(losses) > min_prev_loss + self.cfg.detection_threshold:
            self.cluster_identity = None
        else:
            self.cluster_identity = np.array(losses).argmin(0)


class FedDriftServer(IFCAServer):
    def __init__(self, cfg):
        super().__init__(cfg, 1)

    def merge_clusters(self, clients):
        """Merge clusters whose loss-based distance is below the detection threshold."""
        num_models = len(self.global_models)
        threshold = self.cfg.detection_threshold
        loss_matrix = np.zeros((num_models, num_models))
        for i in range(num_models):
            for j in range(num_models):
                total_size = 0
                for client in clients:
                    if client.cluster_identity == j:
                        data = client.train_set
                        total_size += len(data)
                        loss = average_loss(self.global_models[i], data, client.criterion, self.device)
                        loss_matrix[i][j] += loss * len(data)
                if total_size != 0:
                    loss_matrix[i][j] /= total_size
                else:
                    loss_matrix[i][j] = -1

        distances = np.zeros((num_models, num_models))
        for i in range(num_models):
            for j in range(i, num_models):
                if loss_matrix[i][j] == -1 or loss_matrix[j][i] == -1:
                    dist = -1
                else:
                    dist = max(loss_matrix[i][j] - loss_matrix[i][i], loss_matrix[j][i] - loss_matrix[j][j], 0)
                distances[i][j] = distances[j][i] = dist

        deleted = []
        while True:
            cluster_size = np.zeros(num_models)
            for client in clients:
                if client.cluster_identity is not None:
                    cluster_size[client.cluster_identity] += len(client.train_set)
            cluster_i, cluster_j, min_distance = 0, 0, threshold
            for i in range(num_models):
                for j in range(i + 1, num_models):
                    if distances[i][j] == -1:
                        continue
                    if distances[i][j] < min_distance:
                        cluster_i, cluster_j, min_distance = i, j, distances[i][j]
            if min_distance == threshold:
                break
            size_i, size_j = cluster_size[cluster_i], cluster_size[cluster_j]
            params_i = parameters_to_vector(self.global_models[cluster_i].parameters())
            params_j = parameters_to_vector(self.global_models[cluster_j].parameters())
            merged = (size_i * params_i + size_j * params_j) / (size_i + size_j)
            print(f"[feddrift] merge cluster {cluster_i} and cluster {cluster_j}", flush=True)
            vector_to_parameters(merged, self.global_models[cluster_i].parameters())
            deleted.append(cluster_j)
            for client in clients:
                if client.cluster_identity == cluster_j:
                    client.cluster_identity = cluster_i
            for k in range(num_models):
                if k in (cluster_i, cluster_j):
                    continue
                dist = max(distances[cluster_i][k], distances[cluster_j][k])
                distances[cluster_i][k] = distances[k][cluster_i] = dist
            distances[:, cluster_j] = -1
            distances[cluster_j, :] = -1

        deleted.sort(reverse=True)
        for i in deleted:
            for client in clients:
                if client.cluster_identity is not None and client.cluster_identity > i:
                    client.cluster_identity -= 1
            del self.global_models[i]
