"""Steps shared by all methods: data preparation, client setup, evaluation and the run harness."""

import json
import os
import time

from utils.config import build_config, overrides_from, parse_args, run_name
from utils.gen_dataset import distribute_dataset, draw_missing_classes
from utils.seeding import seed_everything


def prepare_data(cfg):
    """Partition the data, then seed all generators (the partition reseeds NumPy per class)."""
    client_train, client_test, global_tests = distribute_dataset(
        cfg.dataset, cfg.data_dir, cfg.num_clients, cfg.dirichlet_alpha, cfg.seed)
    seed_everything(cfg.seed, cfg.deterministic)
    return client_train, client_test, global_tests


def evaluate_personalized(server, clients, selected, global_tests):
    """Accuracy of each client's encoder and own classifier.

    Under partial participation the server first sends its encoder to all
    clients (evaluation only; a client sampled later receives the full model anyway).
    """
    partial = len(selected) < len(clients)
    if partial and server.cfg.evaluation_encoder_broadcast:
        server.send_encoder(clients)
    local = server.local_evaluate(clients)
    if partial:
        return local, server.evaluate_each_client(clients, global_tests)
    return local, server.evaluate_shared_encoder(clients, global_tests)


def setup(cfg, client_class, server_class):
    """Data, clients with missing classes removed, and the server."""
    client_train, client_test, global_tests = prepare_data(cfg)
    clients = [client_class(k, cfg, client_train[k], client_test[k]) for k in range(cfg.num_clients)]
    draw_missing_classes(clients, cfg.missing_prob, cfg.seed, cfg.num_classes)
    server = server_class(cfg)
    server.record_client_sizes(clients)
    return clients, server, global_tests


def evaluate_global_model(server, clients, global_tests):
    """Send the global model to every client, then evaluate it."""
    server.send_params(clients)
    return server.local_evaluate(clients), server.evaluate_shared_model(clients, global_tests)


def evaluate_client_models(server, clients, global_tests):
    """Evaluate every client's own model."""
    return server.local_evaluate(clients), server.evaluate_each_client(clients, global_tests)


def save_state(path, clients, server):
    import torch
    state = {
        "clients": [{k: v.detach().cpu() for k, v in c.model.state_dict().items()} for c in clients],
        "server": {k: v.detach().cpu() for k, v in server.model.state_dict().items()},
    }
    if hasattr(server, "global_models"):
        state["global_models"] = [{k: v.detach().cpu() for k, v in m.state_dict().items()}
                                  for m in server.global_models]
    if hasattr(clients[0], "personal_model"):
        state["personal_models"] = [{k: v.detach().cpu() for k, v in c.personal_model.state_dict().items()}
                                    for c in clients]
    for name in ("anchors", "etf"):
        if getattr(server, name, None) is not None:
            state["anchors"] = getattr(server, name).detach().cpu()
    torch.save(state, path)


def run_experiment(method, train, argv=None):
    """Parse the command line, run ``train(cfg, log)`` and write the outputs."""
    args = parse_args(method, argv)
    cfg = build_config(method, args.config, overrides_from(method, args))
    if cfg.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    out_dir = os.path.join(cfg.output_dir, args.run_name or run_name(cfg))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as handle:
        json.dump(cfg, handle, indent=2, sort_keys=True)
    print(json.dumps(cfg, sort_keys=True), flush=True)

    from utils.metrics import CurveWriter, write_rows
    curve = CurveWriter(os.path.join(out_dir, "curve.csv"), cfg.rounds)
    start = time.time()

    def log(round_index, local_accuracy, test_accuracy):
        curve.append(round_index, local_accuracy, test_accuracy)
        print(f"round {round_index:3d}  test_accuracy={float(test_accuracy):.4f}  "
              f"local_test_accuracy={float(local_accuracy):.4f}  elapsed={time.time() - start:.0f}s",
              flush=True)

    try:
        clients, server, extras = train(cfg, log)
    finally:
        curve.close()
    if extras.get("diagnostics"):
        write_rows(os.path.join(out_dir, "diagnostics.csv"), extras["diagnostics"])
    if args.save_state:
        save_state(os.path.join(out_dir, "final_state.pt"), clients, server)
    print(f"results written to {out_dir}", flush=True)
