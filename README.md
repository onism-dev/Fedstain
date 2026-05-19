# FedStain

Official implementation of **FedStain: Modeling Higher-Order Stain Statistics for Federated Domain Generalization in Computational Pathology**.

FedStain is a federated domain generalization framework for computational pathology. It improves robustness under stain heterogeneity by exchanging higher-order stain statistics, including **skewness** and **kurtosis**, without sharing raw images.

## Highlights

- Federated domain generalization for computational pathology
- Higher-order stain statistics modeling with skewness and kurtosis
- Privacy-preserving training without raw image sharing
- Supports Camelyon17-style and MvMidog-Fed benchmarks
- Includes multiple federated learning and domain generalization baselines

## Supported Methods

| Method | Description |
|---|---|
| **FedStain** | Proposed method |
| FedAvg | Standard federated averaging |
| FedProx | Federated optimization baseline |
| FedSR | Federated domain generalization baseline |
| FedIIR | Federated invariant risk minimization |
| FedADG | Federated adversarial domain generalization |
| GA | Generalized aggregation |
| CCST | Cross-client style transfer |

## Project Structure

```text
FedStain/
├── main.py                 # Training entry
├── requirements.txt
├── algorithm/
│   ├── client/             # Local training methods
│   └── server/             # Server aggregation and evaluation
├── data/
│   ├── dataset.py
│   └── partition_data.py   # Data partition and domain split
├── model/
│   └── models.py           # Backbone and MixStyle modules
└── utils/
```

## Installation

```bash
git clone https://github.com/onism-dev/FedStain.git
cd FedStain

pip install -r requirements.txt
```

## Data Preparation

Place datasets in the following structure:

```text
data/<dataset_name>/raw/<domain_name>/<label>/<image>
```

Example:

```text
data/came/raw/domain_hospital0/0/image.png
data/came/raw/domain_hospital1/1/image.png
```

Supported datasets:

| Dataset | Description |
|---|---|
| `came` | Camelyon17-style multi-hospital domains |
| `midog` | MvMidog-Fed (https://github.com/onism-dev/MvMidog-Fed-dataset) scanner domains |

The MvMidog-Fed dataset is released in a separate repository:

```text
https://github.com/onism-dev/MvMidog-Fed-dataset
```

Please follow the dataset repository instructions to prepare the `midog` benchmark before running experiments.

## Training

Run FedStain on Camelyon17-style domains:

```bash
python main.py FedStain -d came
```

Run FedStain on MvMidog-Fed:

```bash
python main.py FedStain -d midog
```

Run a baseline method:

```bash
python main.py FedAvg -d came
```

Available methods:

```text
FedStain, FedAvg, FedProx, FedSR, FedIIR, FedADG, GA, CCST
```

## Output

Training results are saved to:

```text
out/<Method>/<dataset>/<timestamp>/
```

The output directory contains:

```text
test_accuracy.pkl
test_accuracy.csv
```

## Key Hyperparameters

FedStain-related arguments can be found in:

```text
algorithm/server/fedstain.py
```

Common options:

| Argument | Description | Default |
|---|---|---|
| `--r` | Fraction of local samples used to estimate skewness and kurtosis | `0.1` |
| `--lambda1` | Contrastive representation alignment weight | `0.1` |
| `--lambda2` | JS prediction alignment weight | `0.1` |
| `--p` | MixStyle application probability | `1.0` |

For `came` and `midog`, the default setting uses ResNet50, Adam optimizer, learning rate `1e-4`, 3 communication rounds, and 3 local epochs.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{zhang2026fedstain,
  title={FedStain: Modeling Higher-Order Stain Statistics for Federated Domain Generalization in Computational Pathology},
  author={Zhang, Fengyi and Zhang, Junya and Sun, Wenzhuo},
  journal={arXiv preprint arXiv:2605.14590},
  year={2026}
}
```
