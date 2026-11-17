# FedMCD: Federated Learning with Missing Classes and Concept Drift

Code for FedMCD and the baselines used in the paper *FedMCD: Cloud–Edge Federated Learning with Missing Classes and Concept Drift*.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

## Data

CIFAR-10 and Fashion-MNIST are read from `--data-dir` (default `data/`) and downloaded there by torchvision if absent.

## Example

```bash
python methods/FedMCD.py --config configs/FedMCD.yaml --dataset cifar10 --drift sudden \
    --drift-fraction 1.0 --missing-prob 0.4 --seed 0 --data-dir data
```

Results are written to `outputs/<run-name>/curve.csv` (test accuracy per round). Settings are in `configs/`.

Methods (`methods/<Method>.py` with `configs/<Method>.yaml`): `FedMCD`, `FedAvg`, `FedProx`, `FedGELA`, `FedCCFA` (with `--class-support-mask` for FedCCFA-PCS+mask), `SCAFFOLD`, `Ditto`, `FedRep`, `FedBABU`, `FedPAC`, `pFedMe`, `IFCA`, `FedDrift`, `Flash`, `FedFM`, `AdaptiveFedAvg`; Fed-ISIC2019: `methods/FedISIC2019.py`.

FedMCD ablation flags: `--no-cone-projection`, `--max-cone-radius 0`, `--group-on-classifier`, `--no-prototype-alignment`, `--static-anchors`, `--no-class-aware-averaging`.
