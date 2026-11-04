import math
import time

import numpy as np
import torch
import torch.nn.functional as F


def _iter_train_batches(train_data):
    train_x_left = train_data["train_x_left"]
    train_x_right = train_data["train_x_right"]
    train_y = train_data["train_y"]
    batch_size = train_data["batch_size"]
    for start in range(0, len(train_y), batch_size):
        sample_id = np.arange(start, min(start + batch_size, len(train_y)))
        yield sample_id, train_x_left[sample_id], train_x_right[sample_id], train_y[sample_id].squeeze(dim=-1)


def _simple_kmeans(features, n_clusters, max_iter=10):
    n_samples = features.size(0)
    if n_clusters <= 1 or n_samples <= 1:
        return torch.zeros(n_samples, dtype=torch.long)

    n_clusters = min(n_clusters, n_samples)
    device = features.device
    initial = torch.linspace(0, n_samples - 1, steps=n_clusters, device=device).round().long()
    centers = features[initial].clone()
    labels = torch.zeros(n_samples, dtype=torch.long, device=device)
    for _ in range(max_iter):
        dist = torch.cdist(features, centers, p=2)
        new_labels = torch.argmin(dist, dim=1)

        new_centers = centers.clone()
        for cluster_idx in range(n_clusters):
            mask = new_labels == cluster_idx
            if torch.any(mask):
                new_centers[cluster_idx] = features[mask].mean(dim=0)

        labels = new_labels
        if torch.allclose(new_centers, centers, atol=1e-5):
            break
        centers = F.normalize(new_centers, dim=-1)

    return labels.cpu()


def _projection_split(features, n_clusters):
    n_samples = features.size(0)
    if n_clusters <= 1 or n_samples <= 1:
        return torch.zeros(n_samples, dtype=torch.long)

    center = F.normalize(features.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
    scores = torch.matmul(features, center)
    order = torch.argsort(scores)
    labels = torch.empty(n_samples, dtype=torch.long, device=features.device)
    chunk_size = int(math.ceil(n_samples / n_clusters))
    for cluster_idx, start in enumerate(range(0, n_samples, chunk_size)):
        labels[order[start:start + chunk_size]] = cluster_idx
    return labels.cpu()


def update_subclass_and_temperature(model, train_loader, device, args, epoch=None):
    start_time = time.time()
    was_training = model.training
    model.eval()

    all_ids = []
    all_labels = []
    all_features = []

    from train_test import build_m2u_batch

    graph_cache = train_loader.get("m2umol_graph_cache")
    with torch.no_grad():
        for sample_id, x_left, x_right, y in _iter_train_batches(train_loader):
            if graph_cache is None:
                _, pair_repr = model(x_left, x_right, return_pair_emb=True)
            else:
                left_m2u_graph = build_m2u_batch(x_left, graph_cache)
                right_m2u_graph = build_m2u_batch(x_right, graph_cache)
                _, pair_repr = model(x_left, x_right, left_m2u_graph, right_m2u_graph, return_pair_emb=True)
            z = model.scl_project(pair_repr)
            all_ids.append(torch.from_numpy(sample_id).long())
            all_labels.append(y.cpu().long())
            all_features.append(z.detach().cpu())

    if was_training:
        model.train()

    sample_ids = torch.cat(all_ids, dim=0)
    labels = torch.cat(all_labels, dim=0)
    features = F.normalize(torch.cat(all_features, dim=0).float(), dim=-1)

    num_samples = int(labels.numel())
    num_classes = int(getattr(args, "class_num", int(labels.max().item()) + 1))
    train_subclass_labels = torch.full((num_samples,), -1, dtype=torch.long)

    counts = torch.bincount(labels, minlength=num_classes)
    present_counts = counts[counts > 0]
    target_cluster_size = max(1, int(args.target_cluster_size))
    max_clusters_per_class = max(1, int(args.max_clusters_per_class))

    next_subclass_id = 0
    head_classes = 0
    tail_classes = 0

    for class_id in range(num_classes):
        class_mask = labels == class_id
        class_indices = torch.where(class_mask)[0]
        n_c = int(class_indices.numel())
        if n_c == 0:
            continue

        z_c = features[class_indices]

        if n_c >= 2 * target_cluster_size:
            head_classes += 1
            n_clusters = int(math.ceil(n_c / target_cluster_size))
            n_clusters = min(n_clusters, max_clusters_per_class)
            if n_c * n_clusters > 2000000:
                cluster_labels = _projection_split(z_c, n_clusters)
            else:
                cluster_labels = _simple_kmeans(z_c.to(device), n_clusters)
        else:
            tail_classes += 1
            n_clusters = 1
            cluster_labels = torch.zeros(n_c, dtype=torch.long)

        global_ids = sample_ids[class_indices]
        train_subclass_labels[global_ids] = cluster_labels + next_subclass_id
        next_subclass_id += n_clusters

    missing = int((train_subclass_labels < 0).sum().item())
    elapsed = time.time() - start_time
    stats = present_counts.float() if present_counts.numel() > 0 else torch.tensor([0.0])
    print(
        "[SCL update] "
        "epoch: {} | total samples: {} | total classes: {} | "
        "head classes: {} | tail classes: {} | total subclasses: {} | "
        "class count min/max/mean: {:.0f}/{:.0f}/{:.2f} | "
        "missing subclass labels: {} | clustering time: {:.2f}s".format(
            "NA" if epoch is None else epoch,
            num_samples, int(present_counts.numel()),
            head_classes, tail_classes, next_subclass_id,
            stats.min().item(), stats.max().item(), stats.mean().item(),
            missing, elapsed
        )
    )
    print("[SCL update] class counts:", present_counts.tolist())

    return train_subclass_labels.long()
