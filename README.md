# SAF — Stabilized A-Feedback for Federated LoRA

> *AI-generated README.*

**TL;DR.** Cross-round EMA on LoRA's direction factor `A` only — keeping `B` on the FedAvg path — fixes federated LoRA's heterogeneity collapse without throttling `B`'s learning rate. One line of code, no new hyperparameters.

## Method

LoRA writes the weight update as `ΔW = BA`. Under high data heterogeneity (Dirichlet `α ≤ 0.1`), FedAvg accumulates per-round adapter dispersion and converges to the majority class. Feeding **both** factors through a cross-round EMA squeezes `B`'s effective learning rate by `(1−β)`; feeding **neither** leaves a non-vanishing stationary EMA gap. SAF resolves the asymmetry: feed `A` back only, keep `B` on the FedAvg path during training, and average `B̄_T` once at deployment.

## Install

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the environment of the most recent runs (Python 3.13, RTX 3090 with 24 GB). Those runs used the CUDA 12.6 builds of torch and torchvision; to install them:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt
```

## Run

```bash
python run.py --config <path/to/config.yaml> --gpu 0
```

Example — SAF on MNLI / FedIT at α = 0.05, seed 42:

```bash
python run.py --config configs/paper/s82/mnli/mnli_fedit_saf09_a005_s42.yaml --gpu 0
```

## The five EMA configurations

| Configuration | `training.ema_decay` | `training.ema_feedback` |
|---|---|---|
| Vanilla       | `0.0` | (omit)   |
| EMA           | `0.9` | `full`   |
| EMA-Eval      | `0.9` | `none`   |
| EMA-B         | `0.9` | `b_only` |
| **SAF** (ours) | `0.9` | `a_only` |

Two further options serve as controls. `training.ema_feedback: ab_only` feeds back `A` and `B` but not the classifier head. `evaluation.hybrid_endpoints: true` also evaluates, at every evaluation round, models that take only some tensors from the EMA and the rest from the training path (`hyb_a`: `A`; `hyb_ab`: `A` and `B`; `hyb_a_head`: `A` and the head); they are logged as extra test splits and do not change the training trajectory.

## Outputs

`results/<dataset>/<method>/<alpha>/<config>/<run_name>/` — metrics, per-round JSON logs, and plots if generated.
