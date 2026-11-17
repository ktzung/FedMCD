"""Fed-ISIC2019 case study (FedAvg, FedProx, FedGELA, FedMCD).

    python methods/FedISIC2019.py --prepare-only
    python methods/FedISIC2019.py --method fedmcd --seed 0 --rounds 200
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

from entities.FedISIC2019 import (ANCHOR_UPDATE_EVERY, ANCHOR_WARMUP_ROUNDS, CLASSIFIER_UPDATE_FRACTION, FEDGELA_TEMPERATURE,
                                  FEDPROX_MU, MAX_CONE_RADIUS, PROTOTYPE_START_ROUND, PROTOTYPE_TEMPERATURE,
                                  RESIDUAL_PENALTY, aggregate, anchor_update, average_encoders, average_global_prototypes, class_balance_weights,
                                  finite_state, group_and_average, local_prototypes, next_batch, predict_global,
                                  predict_with_head, prototype_weight)
from utils.anchors import simplex_etf
from utils.cone import cone_radius, project_to_cone
from utils.isic import (FEATURE_DIM, INIT_STATE_NAME, LOCAL_UPDATES, NUM_CLASSES, NUM_CLIENTS, MetricsWriter, classifier,
                        cpu_state, features, flamby, make_model, metric_triplet, optimizer_to_cpu, optimizer_to_device,
                        prepare_initialization, seed_all, warmup_factor)


def run_baseline(method, seed, rounds, updates, workers, init_path, metrics_path, log=print):
    isic = flamby()
    seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_sets = [isic.FedIsic2019(center=i, train=True, pooled=False) for i in range(NUM_CLIENTS)]
    test_sets = [isic.FedIsic2019(center=i, train=False, pooled=False) for i in range(NUM_CLIENTS)]
    pooled_test = isic.FedIsic2019(train=False, pooled=True)
    generators = [torch.Generator().manual_seed(seed * 1000 + i) for i in range(NUM_CLIENTS)]
    train_loaders = [DataLoader(d, batch_size=isic.BATCH_SIZE, shuffle=True, num_workers=workers,
                                pin_memory=True, generator=g) for d, g in zip(train_sets, generators)]
    train_iterators = [iter(loader) for loader in train_loaders]
    center_tests = [DataLoader(d, batch_size=isic.BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True)
                    for d in test_sets]
    pooled_loader = DataLoader(pooled_test, batch_size=isic.BATCH_SIZE, shuffle=False, num_workers=workers,
                               pin_memory=True)
    counts = [torch.from_numpy(np.bincount(np.asarray(d.targets, dtype=np.int64), minlength=NUM_CLASSES)).float()
              for d in train_sets]
    sizes = [len(d) for d in train_sets]

    global_model = make_model(init_path, device)
    seed_all(seed)
    anchors = simplex_etf(NUM_CLASSES, FEATURE_DIM, "cpu").to(device)
    if method == "fedgela":
        with torch.no_grad():
            classifier(global_model).weight.copy_(anchors)
            classifier(global_model).bias.zero_()
        classifier(global_model).weight.requires_grad_(False)
        classifier(global_model).bias.requires_grad_(False)

    writer = MetricsWriter(metrics_path)
    optimizer_states = [None] * NUM_CLIENTS
    for round_index in range(rounds):
        global_state = cpu_state(global_model)
        local_states = []
        for client_id in range(NUM_CLIENTS):
            local = make_model(init_path, device)
            local.load_state_dict(global_state)
            if method == "fedgela":
                classifier(local).weight.requires_grad_(False)
                classifier(local).bias.requires_grad_(False)
            trainable = [p for p in local.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable, lr=isic.LR)
            if optimizer_states[client_id] is not None:
                optimizer.load_state_dict(optimizer_states[client_id])
                optimizer_to_device(optimizer, device)
            loss_function = isic.BaselineLoss().to(device)
            phi = class_balance_weights(counts[client_id], NUM_CLASSES, device)
            local.train()
            for _ in range(updates):
                try:
                    inputs, labels = next(train_iterators[client_id])
                except StopIteration:
                    train_iterators[client_id] = iter(train_loaders[client_id])
                    inputs, labels = next(train_iterators[client_id])
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if method == "fedgela":
                    logits = (features(local, inputs) @ anchors.T) * phi.unsqueeze(0) / FEDGELA_TEMPERATURE
                else:
                    logits = local(inputs)
                loss = loss_function(logits, labels)
                if method == "fedprox":
                    prox = sum((p - global_state[name].to(device)).pow(2).sum() for name, p in local.named_parameters())
                    loss = loss + 0.5 * FEDPROX_MU * prox
                loss.backward()
                optimizer.step()
            optimizer_states[client_id] = optimizer_to_cpu(optimizer.state_dict())
            local_states.append(cpu_state(local))
            del local, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        global_model.load_state_dict(aggregate(local_states, sizes))
        if method == "fedgela":
            with torch.no_grad():
                classifier(global_model).weight.copy_(anchors)
                classifier(global_model).bias.zero_()

        pooled = metric_triplet(*predict_global(global_model, pooled_loader, device, method, anchors))
        center_balanced = [metric_triplet(*predict_global(global_model, loader, device, method, anchors))[0]
                           for loader in center_tests]
        writer.write(round_index, pooled, center_balanced)
        log(f"round {round_index:3d}  pooled_balanced_accuracy={pooled[0]:.4f}")
    writer.close()
    return global_model, {"anchors": anchors}


def run_fedmcd_isic(seed, rounds, updates, workers, init_path, metrics_path, log=print):
    if updates < 2:
        raise ValueError("updates must be at least 2 (classifier and encoder steps)")
    isic = flamby()
    seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_sets = [isic.FedIsic2019(center=i, train=True, pooled=False) for i in range(NUM_CLIENTS)]
    test_sets = [isic.FedIsic2019(center=i, train=False, pooled=False) for i in range(NUM_CLIENTS)]
    pooled_test = isic.FedIsic2019(train=False, pooled=True)
    generators = [torch.Generator().manual_seed(seed * 1000 + i) for i in range(NUM_CLIENTS)]
    train_loaders = [DataLoader(d, batch_size=isic.BATCH_SIZE, shuffle=True, num_workers=workers,
                                pin_memory=True, generator=g) for d, g in zip(train_sets, generators)]
    prototype_loaders = [DataLoader(d, batch_size=isic.BATCH_SIZE, shuffle=False, num_workers=workers,
                                    pin_memory=True) for d in train_sets]
    train_iterators = [iter(loader) for loader in train_loaders]
    center_tests = [DataLoader(d, batch_size=isic.BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True)
                    for d in test_sets]
    pooled_loader = DataLoader(pooled_test, batch_size=isic.BATCH_SIZE, shuffle=False, num_workers=workers,
                               pin_memory=True)
    counts = [torch.from_numpy(np.bincount(np.asarray(d.targets, dtype=np.int64), minlength=NUM_CLASSES)).float()
              for d in train_sets]
    presences = [(value > 0).to(device) for value in counts]
    sizes = [len(d) for d in train_sets]

    global_model = make_model(init_path, device)
    initial_anchors = simplex_etf(NUM_CLASSES, FEATURE_DIM, "cpu").to(device)
    anchors = initial_anchors.clone()
    with torch.no_grad():
        classifier(global_model).weight.copy_(anchors)
        classifier(global_model).bias.zero_()
    personal_weights = [anchors.clone() for _ in range(NUM_CLIENTS)]
    personal_biases = [torch.zeros(NUM_CLASSES, device=device) for _ in range(NUM_CLIENTS)]
    global_prototypes = anchors.clone()
    classifier_states = [None] * NUM_CLIENTS
    encoder_states = [None] * NUM_CLIENTS
    classifier_steps = max(1, int(round(updates * CLASSIFIER_UPDATE_FRACTION)))
    encoder_steps = updates - classifier_steps

    writer = MetricsWriter(metrics_path, extra_fields=["anchor_updated", "num_groups", "rejected_clients"])
    for round_index in range(rounds):
        global_state = cpu_state(global_model)
        states, weights, biases, prototypes, accepted_sizes, accepted_presences, accepted_ids = [], [], [], [], [], [], []
        rejected = 0
        for client_id in range(NUM_CLIENTS):
            local = make_model(init_path, device)
            local.load_state_dict(global_state)
            with torch.no_grad():
                classifier(local).weight.copy_(personal_weights[client_id])
                classifier(local).bias.copy_(personal_biases[client_id])
            radii, _ = cone_radius(counts[client_id], NUM_CLASSES, max_radius=MAX_CONE_RADIUS,
                                   warmup=warmup_factor(round_index, rounds), device=device)
            loss_function = isic.BaselineLoss().to(device)

            # Step 2: local classifier with cone projection after every update
            for p in local.parameters():
                p.requires_grad_(False)
            for p in classifier(local).parameters():
                p.requires_grad_(True)
            optimizer = torch.optim.Adam(classifier(local).parameters(), lr=isic.LR)
            if classifier_states[client_id] is not None:
                optimizer.load_state_dict(classifier_states[client_id])
                optimizer_to_device(optimizer, device)
            for _ in range(classifier_steps):
                (inputs, labels), train_iterators[client_id] = next_batch(train_iterators[client_id],
                                                                          train_loaders[client_id])
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(local(inputs), labels)
                residual = classifier(local).weight - anchors
                loss = loss + 0.5 * RESIDUAL_PENALTY * residual.pow(2).sum()
                loss.backward()
                optimizer.step()
                project_to_cone(classifier(local).weight, anchors, radii, bias=classifier(local).bias)
            classifier_states[client_id] = optimizer_to_cpu(optimizer.state_dict())

            # Step 3: encoder training with prototype alignment
            for p in local.parameters():
                p.requires_grad_(True)
            for p in classifier(local).parameters():
                p.requires_grad_(False)
            encoder_parameters = [p for p in local.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(encoder_parameters, lr=isic.LR)
            if encoder_states[client_id] is not None:
                optimizer.load_state_dict(encoder_states[client_id])
                optimizer_to_device(optimizer, device)
            weight = prototype_weight(counts[client_id])
            for _ in range(encoder_steps):
                (inputs, labels), train_iterators[client_id] = next_batch(train_iterators[client_id],
                                                                          train_loaders[client_id])
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                values = features(local, inputs)
                loss = loss_function(F.linear(values, classifier(local).weight, classifier(local).bias), labels)
                if round_index >= PROTOTYPE_START_ROUND and weight > 0.0:
                    logits = F.normalize(values, dim=1) @ F.normalize(global_prototypes, dim=1).T
                    loss = loss + weight * F.cross_entropy(logits / PROTOTYPE_TEMPERATURE, labels)
                loss.backward()
                optimizer.step()
            encoder_states[client_id] = optimizer_to_cpu(optimizer.state_dict())

            state = cpu_state(local)
            if not finite_state(state):
                rejected += 1
            else:
                states.append(state)
                weights.append(classifier(local).weight.detach().clone())
                biases.append(classifier(local).bias.detach().clone())
                prototypes.append(local_prototypes(local, prototype_loaders[client_id], device))
                accepted_sizes.append(sizes[client_id])
                accepted_presences.append(presences[client_id])
                accepted_ids.append(client_id)
            del local, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not states:
            raise RuntimeError("all clients returned non-finite models")

        global_model.load_state_dict(average_encoders(states, accepted_sizes))
        updated = False
        if round_index >= ANCHOR_WARMUP_ROUNDS and round_index % ANCHOR_UPDATE_EVERY == 0:
            anchors, updated = anchor_update(initial_anchors, anchors, prototypes)
        global_prototypes = average_global_prototypes(global_prototypes, prototypes)
        merged = group_and_average(weights, biases, accepted_presences, anchors)
        for position, client_id in enumerate(accepted_ids):
            personal_weights[client_id] = merged.weights[position]
            personal_biases[client_id] = merged.biases[position]
            missing = ~presences[client_id]
            personal_weights[client_id][missing] = anchors[missing]
            personal_biases[client_id][missing] = 0.0
        with torch.no_grad():
            classifier(global_model).weight.copy_(merged.global_weight)
            classifier(global_model).bias.copy_(merged.global_bias)
        project_to_cone(classifier(global_model).weight, anchors,
                        torch.full((NUM_CLASSES,), MAX_CONE_RADIUS, device=device), bias=classifier(global_model).bias)

        pooled = metric_triplet(*predict_with_head(global_model, pooled_loader, device,
                                         classifier(global_model).weight, classifier(global_model).bias))
        center_balanced = []
        for client_id, loader in enumerate(center_tests):
            truth, pred = predict_with_head(global_model, loader, device, personal_weights[client_id], personal_biases[client_id])
            center_balanced.append(float(balanced_accuracy_score(truth, pred)))
        writer.write(round_index, pooled, center_balanced, anchor_updated=int(updated),
                     num_groups=merged.num_groups, rejected_clients=rejected)
        log(f"round {round_index:3d}  pooled_balanced_accuracy={pooled[0]:.4f}  groups={merged.num_groups}")
    writer.close()
    return global_model, {"anchors": anchors, "personal_weights": personal_weights, "personal_biases": personal_biases}


METHODS = ("fedavg", "fedprox", "fedgela", "fedmcd")


def main(argv=None):
    parser = argparse.ArgumentParser(description="FedMCD and baselines on Fed-ISIC2019 (FLamby).")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--updates", type=int, default=LOCAL_UPDATES, help="local mini-batch updates per round")
    parser.add_argument("--workers", type=int, default=0, help="data loader workers")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "isic"))
    parser.add_argument("--init-state", default=None,
                        help=f"pretrained initial weights (default: OUTPUT_DIR/{INIT_STATE_NAME})")
    parser.add_argument("--prepare-only", action="store_true",
                        help="download the ImageNet-pretrained EfficientNet-B0 and save it as the initial state")
    parser.add_argument("--save-state", action="store_true")
    args = parser.parse_args(argv)

    init_path = args.init_state or os.path.join(args.output_dir, INIT_STATE_NAME)
    if args.prepare_only:
        prepare_initialization(init_path)
        print(f"initial state ready: {init_path}")
        return
    if args.method is None:
        parser.error("--method is required unless --prepare-only is given")
    run_dir = os.path.join(args.output_dir, f"{args.method}_s{args.seed}")
    metrics_path = os.path.join(run_dir, "metrics.csv")
    if args.method == "fedmcd":
        model, extras = run_fedmcd_isic(args.seed, args.rounds, args.updates, args.workers, init_path, metrics_path)
    else:
        model, extras = run_baseline(args.method, args.seed, args.rounds, args.updates, args.workers,
                                     init_path, metrics_path)
    if args.save_state:
        state = {"global_model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        for key, value in extras.items():
            state[key] = [v.detach().cpu() for v in value] if isinstance(value, list) else value.detach().cpu()
        torch.save(state, os.path.join(run_dir, "final_state.pt"))
    print(f"results written to {run_dir}")


if __name__ == "__main__":
    main()
