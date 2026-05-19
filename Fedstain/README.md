# FedStain

Official implementation of **FedStain: Modeling Higher-Order Stain Statistics for Federated Domain Generalization in Computational Pathology**.

FedStain exchanges **skewness** and **kurtosis** (higher-order stain moments) across clients in federated learning, together with cross-site contrastive aggregation, to improve domain generalization under stain heterogeneity—without sharing raw pixels.

## Supported algorithms

| Algorithm | Role |
|-----------|------|
| **FedStain** | Proposed method |
| FedAvg | Federated learning baseline |
| FedProx | Federated optimization baseline |
| FedSR | Federated domain generalization baseline |
| FedIIR | Invariant risk minimization (federated) |
| FedADG | Adversarial DG (federated) |
| GA | Generalized aggregation |
| CCST | Cross-client style transfer |

## Project layout

```
FedStain/
├── main.py                 # Training entry (leave-one-domain-out)
├── requirements.txt
├── algorithm/
│   ├── client/             # Per-algorithm local training
│   └── server/             # Federated aggregation & evaluation
├── data/
│   ├── dataset.py
│   └── partition_data.py   # Federated partition & domain split
├── model/
│   └── models.py           # Backbone + MixStyle (higher-order statistics)
└── utils/
```

## Data preparation

Place datasets under `data/<dataset_name>/raw/<domain_name>/<label>/<image>`.

Supported pathology benchmarks in this release:

- **came** — Camelyon17-style multi-hospital domains (`domain_hospital0` … `domain_hospital4`)
- **midog** — MvMidog-Fed scanner domains (`Hamamatsu XR`, `3D Histech`, `Aperio CS2`, `Hamamatsu S360`)

General DG datasets (`pacs`, `vlcs`, `office_home`) are also supported for baseline reproduction.

## Training

```bash
pip install -r requirements.txt

# FedStain on Camelyon17 (leave-one-hospital-out)
python main.py FedStain -d came

# FedStain on MvMidog-Fed
python main.py FedStain -d midog

# Baseline example
python main.py FedAvg -d came
```

Results are written to `out/<Algorithm>/<dataset>/<timestamp>/`, including per-domain `test_accuracy.pkl` and a summary `test_accuracy.csv`.

### Hyper-parameters (FedStain)

Key arguments (see `algorithm/server/fedstain.py`):

- `--r` — fraction of local samples used to estimate skewness/kurtosis (default `0.1`)
- `--lambda1` — contrastive representation alignment weight (default `0.1`)
- `--lambda2` — JS prediction alignment weight (default `0.1`)
- `--p` — MixStyle application probability (default `1.0`)

For `came` / `midog`, `main.py` sets `ResNet50`, Adam, `lr=1e-4`, 3 communication rounds, and 3 local epochs by default.

## Citation

If you use this repository or find our code helpful in your research, please cite our paper:

> Fengyi Zhang, Junya Zhang, and Wenzhuo Sun. **FedStain: Modeling Higher-Order Stain Statistics for Federated Domain Generalization in Computational Pathology.** *arXiv preprint arXiv:2605.14590*, 2026.

```bibtex
@article{zhang2026fedstain,
  title={FedStain: Modeling Higher-Order Stain Statistics for Federated Domain Generalization in Computational Pathology},
  author={Zhang, Fengyi and Zhang, Junya and Sun, Wenzhuo},
  journal={arXiv preprint arXiv:2605.14590},
  year={2026}
}
```
