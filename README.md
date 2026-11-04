# CMSCL

Implementation accompanying **Cross-Modal Knowledge Transfer and Subclass-Aware Contrastive Learning for Drug-Drug Interaction Event Prediction**.

CMSCL combines molecular representations enriched with transferred geometric, textual, and biological knowledge, biomedical knowledge graph representations, and subclass-aware contrastive learning (SCL) for DDI event classification.

## Repository structure

```text
main.py                         Training and five-fold evaluation entry point
train_test.py                   Training and evaluation routines
model/CMSCL.py                  Combined CMSCL model
model/m2umol_encoder.py         Molecular encoder and pretrained weight loading
model/hgnn.py                   Knowledge graph encoder
utils/scl_loss.py              SCL loss used by the training entry point
utils/scl_cluster.py           Training subclass discovery
scripts/build_m2umol_graph_cache.py
requirements.txt               Dependency names, not a locked experiment environment
```


## Environment

The selected PyTorch build is **2.2.1+cu121** (CUDA 12.1).

`requirements.txt` pins this PyTorch build; Python, DGL, PyG and other library versions still need to be recorded from the actual experiment environment. Select DGL and PyG builds compatible with PyTorch 2.2.1 and CUDA 12.1. The full dependency combination has not been validated here. `dgllife` is used by the legacy molecular module.

Run all commands below from the directory containing `main.py`. The code uses CUDA device 0 when available and CPU otherwise; CPU training performance has not been validated here.

## Data and pretrained files

The Deng and Ryu TSV inputs and the M2UMol-compatible `data/symbo.npy` are included in this repository. Each dataset directory contains `nodes.tsv` (node ID, name, type), `edges.tsv` (head ID, relation ID, tail ID), `smiles.tsv` (drug ID, SMILES), and `ddi.tsv` (left drug ID, right drug ID, event label). Obtain the M2UMol pretrained weights separately and place them at:

```text
pre-trained_M2UMol.pt
```

The pretrained weights originate from M2UMol and are used in this project for molecular encoding.

- File: `pre-trained_M2UMol.pt`
- Download: [Baidu Netdisk](https://pan.baidu.com/s/1SEg3y_l4xTAy-NOoJFONQw?pwd=wm6u)
- Extraction code: `wm6u`

Download the file and place it in the project root, alongside `main.py`. Exact upstream revisions still need to be documented. Pretrained weights, trained checkpoints and generated caches are excluded from Git; the documented TSV inputs and symbol lookup file are included.

Build molecular graph caches:

```bash
python scripts/build_m2umol_graph_cache.py --ddi_names Deng Ryu --symbo_path ./data/symbo.npy
```

For a single prepared dataset, pass only its name to `--ddi_names`. Check the reported failed-drug count before training.

## Training and evaluation

Create the log directory before the first run:

```bash
python -c "from pathlib import Path; Path('log').mkdir(exist_ok=True)"
```

Example commands for the combined model with SCL enabled:

```bash
python main.py --ddi_name Deng --condition S1 --mode concat --use_scl
python main.py --ddi_name Ryu --condition S1 --mode concat --use_scl
```

Use `--condition S2` or `--condition S3` for the corresponding cold-start test settings. These are commands for the current implementation, not independently verified reproductions of the manuscript tables.

Available representation modes:

| Mode | Representations |
| --- | --- |
| `concat` | Knowledge graph + M2UMol molecular features |
| `kg` | Knowledge graph only |
| `use_m2umol_online` | M2UMol molecular features only |

Disable SCL for its ablation:

```bash
python main.py --ddi_name Deng --condition S1 --mode concat --no_scl
```

These mode switches describe implemented variants; they should not automatically be equated with every ablation in the manuscript.

### Current defaults

| Parameter | Default |
| --- | --- |
| Dataset / setting / mode | Deng / S3 / concat |
| Folds / batch size | 5 / 4096 |
| Maximum epochs / patience | 1000 / 30 |
| Hidden dimension / HGNN layers | 256 / 3 |
| SCL enabled | Yes |
| `scl_lambda` / `scl_beta` | 0.01 / 0.5 |
| `scl_tau1` (subclass) / `scl_tau2` (event) | 0.07 / 0.2 |
| Warm-up epochs / cluster update interval | 10 / 5 |
| Target subclass size / maximum subclasses per event | 128 / 20 |
| Projection dimension | 128 |

| Setting | Learning rate | Weight decay | M2UMol learning rate |
| --- | --- | --- | --- |
| S1 | 0.0002 | 0.0005 | 0.0001 |
| S2 | 0.0001 | 0.0005 | 0.00005 |
| S3 | 0.0001 | 0.001 | 0.00001 |

Explicit command-line learning rates override these setting-dependent defaults. See `python main.py --help` after installing dependencies.

## Outputs and reproducibility notes

The program reports fold-level test scores and the mean and standard deviation across folds for accuracy, F1, precision, recall, and Cohen's kappa. Checkpoints are saved under `data/<KG>+<dataset>/<condition>/Fold<i>/`; text logs are saved under `log/`.

Runs with the same dataset, mode, condition, hidden dimension and layer count overwrite the same log path. Checkpoint paths also overlap across modes and SCL configurations. Preserve outputs between experiments.

The entry point seeds NumPy and PyTorch with 42. Data splitting also uses seed 42, but deterministic GPU execution is not enforced. Validation uses a stratified pair-level split of the training portion; rare classes with fewer than two samples are retained in training.

The subclass-count rule follows the target-size threshold in the manuscript. One implementation detail requires reconciliation: when `n_c * n_clusters > 2000000`, subclass discovery uses projection-based grouping instead of K-means. No claim of exact manuscript reproduction is made for this unvalidated release.

## Acknowledgements and code provenance

This project builds on HetDDI, **HetDDI: a pre-trained heterogeneous graph neural network model for drug-drug interaction prediction**, and uses M2UMol-related molecular encoding and pretrained resources, **Multi-to-uni modal knowledge transfer pre-training for molecular representation learning**. Please cite these works when using their contributions.

The framework in this repository incorporates SCL and the current DDI training pipeline. Exact upstream repository URLs, revisions, and a file-level attribution review remain to be documented. This acknowledgement does not replace any upstream copyright notices or terms applying to code, data, or weights. No LICENSE file is added in this preparation step.
