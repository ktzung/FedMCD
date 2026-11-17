"""Experiment configuration: YAML files merged in order, then command-line overrides.

configs/common.yaml and configs/<Method>.yaml are read first, then every --config file.
"""

import argparse
import os

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
CONFIG_FILES = {
    "fedmcd": "FedMCD.yaml", "fedavg": "FedAvg.yaml", "fedprox": "FedProx.yaml", "fedgela": "FedGELA.yaml",
    "fedccfa": "FedCCFA.yaml", "scaffold": "SCAFFOLD.yaml", "ditto": "Ditto.yaml", "fedrep": "FedRep.yaml",
    "fedbabu": "FedBABU.yaml", "fedpac": "FedPAC.yaml", "pfedme": "pFedMe.yaml", "ifca": "IFCA.yaml",
    "feddrift": "FedDrift.yaml", "flash": "Flash.yaml", "fedfm": "FedFM.yaml", "adaptive_fedavg": "AdaptiveFedAvg.yaml",
}
METHODS = tuple(CONFIG_FILES)


class Config(dict):
    """A dict whose keys can also be read as attributes."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(f"missing config key {key!r}") from error

    def __setattr__(self, key, value):
        self[key] = value


def load_yaml(path):
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def default_config_files(method):
    return [os.path.join(CONFIG_DIR, "common.yaml"), os.path.join(CONFIG_DIR, CONFIG_FILES[method])]


def parse_value(text):
    """Parse a ``--set key=value`` value with YAML rules (numbers, booleans, null)."""
    return yaml.safe_load(text)


def build_config(method, extra_files=(), overrides=None):
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    cfg = Config()
    for path in list(default_config_files(method)) + list(extra_files):
        cfg.update(load_yaml(path))
    for key, value in (overrides or {}).items():
        if value is not None:
            cfg[key] = value
    cfg.method = method
    validate(cfg)
    return cfg


def validate(cfg):
    if cfg.drift not in ("none", "sudden", "incremental", "recurrent"):
        raise ValueError(f"unknown drift {cfg.drift!r}")
    if not 0.0 <= float(cfg.drift_fraction) <= 1.0:
        raise ValueError("drift_fraction must be in [0, 1]")
    if not 0.0 <= float(cfg.missing_prob) <= 1.0:
        raise ValueError("missing_prob must be in [0, 1]")
    if not 0.0 < float(cfg.participation) <= 1.0:
        raise ValueError("participation must be in (0, 1]")
    if cfg.method == "fedmcd":
        if cfg.balanced_epochs < 1:
            raise ValueError("FedMCD needs at least one balanced-classifier epoch")
        if not 0.0 <= cfg.anchor_gram_weight <= 1.0 or not 0.0 <= cfg.anchor_ema < 1.0:
            raise ValueError("anchor_gram_weight must be in [0, 1] and anchor_ema in [0, 1)")


def run_name(cfg):
    variant = ""
    if cfg.method == "fedccfa" and cfg.class_support_mask:
        variant = "-mask"
    if cfg.method == "fedmcd":
        for key, name, default in ABLATION_SWITCHES:
            if cfg[key] != default:
                variant += f"-{name}" if key != "max_cone_radius" or cfg[key] == 0 else f"-radius{cfg[key]:g}"
    drift = cfg.drift if cfg.drift == "none" else f"{cfg.drift}{float(cfg.drift_fraction):g}"
    return (f"{cfg.method}{variant}_{cfg.dataset}_K{cfg.num_clients}q{float(cfg.participation):g}"
            f"_{drift}_p{float(cfg.missing_prob):g}_s{cfg.seed}")


# (config key, run-name tag, value of full FedMCD)
ABLATION_SWITCHES = (
    ("anchor_update", "static-anchors", True),
    ("max_cone_radius", "anchor-only", 0.25),
    ("cone_projection", "no-cone-projection", True),
    ("group_on_residuals", "group-on-classifier", True),
    ("prototype_alignment", "no-prototype-alignment", True),
    ("class_aware_averaging", "no-class-aware-averaging", True),
)


def parse_args(method, argv=None):
    parser = argparse.ArgumentParser(description=f"{method}: federated learning under concept drift and missing classes.")
    parser.add_argument("--dataset", choices=("cifar10", "fashion_mnist"))
    parser.add_argument("--drift", choices=("none", "sudden", "incremental", "recurrent"))
    parser.add_argument("--drift-fraction", type=float, help="fraction delta of drifting clients")
    parser.add_argument("--missing-prob", type=float, help="per-class removal probability p_miss")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--clients", type=int, dest="num_clients")
    parser.add_argument("--participation", type=float, help="fraction of clients sampled per round")
    parser.add_argument("--config", action="append", default=[],
                        help="extra YAML file merged after the defaults (repeatable)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override any config key (repeatable)")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-name", help="output sub-directory (default: derived from the settings)")
    parser.add_argument("--device")
    parser.add_argument("--deterministic", action="store_true", default=None,
                        help="deterministic kernels; bit-identical repeats on the same hardware")
    parser.add_argument("--max-grad-norm", type=float, help="clip the gradient norm of every local step")
    parser.add_argument("--save-state", action="store_true",
                        help="save the final client and server models to final_state.pt")

    if method == "fedmcd":
        add_fedmcd_arguments(parser)
    if method == "fedccfa":
        add_fedccfa_arguments(parser)
    return parser.parse_args(argv)


def add_fedmcd_arguments(parser):
    ablation = parser.add_argument_group("FedMCD component ablation (one at a time)")
    ablation.add_argument("--static-anchors", action="store_true",
                          help="keep the initial ETF anchors (no Step 4)")
    ablation.add_argument("--max-cone-radius", type=float,
                          help="rho_max; 0 keeps every classifier row at its anchor (no residual learning)")
    ablation.add_argument("--no-cone-projection", action="store_true",
                          help="unconstrained residuals: only missing-class rows are reset to the anchor")
    ablation.add_argument("--group-on-classifier", action="store_true",
                          help="Step 5 on the full classifier rows instead of the residuals")
    ablation.add_argument("--no-prototype-alignment", action="store_true", help="no prototype loss in Step 3")
    ablation.add_argument("--no-class-aware-averaging", action="store_true",
                          help="Step 6 averages over all group members, also those without the class")
    ablation.add_argument("--diagnostics", action="store_true", default=None,
                          help="write per-round residual and grouping diagnostics")



def add_fedccfa_arguments(parser):
    fedccfa = parser.add_argument_group("FedCCFA")
    fedccfa.add_argument("--class-support-mask", action="store_true", default=None,
                         help="FedCCFA-PCS+mask (default with missing classes: FedCCFA-PCS)")


def overrides_from(method, args):
    overrides = {
        "dataset": args.dataset, "drift": args.drift, "drift_fraction": args.drift_fraction,
        "missing_prob": args.missing_prob, "seed": args.seed, "rounds": args.rounds,
        "num_clients": args.num_clients, "participation": args.participation,
        "data_dir": args.data_dir, "output_dir": args.output_dir, "device": args.device,
        "deterministic": args.deterministic, "max_grad_norm": args.max_grad_norm,
    }
    if method == "fedmcd":
        overrides.update({
            "anchor_update": False if args.static_anchors else None,
            "max_cone_radius": args.max_cone_radius,
            "cone_projection": False if args.no_cone_projection else None,
            "group_on_residuals": False if args.group_on_classifier else None,
            "prototype_alignment": False if args.no_prototype_alignment else None,
            "class_aware_averaging": False if args.no_class_aware_averaging else None,
            "diagnostics": args.diagnostics,
        })
    if method == "fedccfa":
        overrides["class_support_mask"] = args.class_support_mask
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key.strip()] = parse_value(value)
    return overrides
