# DDI-FP-Graph

Updated training pipelines for the paper [Molecular Fingerprints Are a Simple Yet Effective Solution to the Drug–Drug Interaction Problem](https://icml-compbio.github.io/2022/papers/WCBICML2022_paper_72.pdf).

## What's new?

- **Modern PyTorch Lightning workflows** live under [`GPU/`](GPU) with W&B integration and Bayesian sweeps, now using the unified `lightning.pytorch` API.
- **TPU-ready TensorFlow GNN pipeline** under [`TPU/`](TPU) for converting the dataset and running on modern TPUs, updated for TensorFlow 2.15 and TF-GNN 1.0.3.
- **Symmetric fingerprint fusion** for the baseline models on both PyTorch and TPU stacks, combining union/intersection/exclusive fingerprints and post-encoder interactions that remain invariant to swapping the drug order.
- **Reproducible environments** via the provided [`pyproject.toml`](pyproject.toml) and [`Dockerfile`](Dockerfile).

## Quickstart

Install dependencies with Poetry (Python 3.10 through 3.12 are supported with the current TensorFlow stack):

```bash
poetry install
```

Train a graph model with PyTorch Lightning and log to Weights & Biases:

```bash
python -m GPU.train --config GPU/configs/graph.yaml --run-name dev-run
```

To use the Bayesian sweep configuration:

```bash
wandb sweep GPU/sweeps/graph_bayesian.yaml
wandb agent <entity/project>/<sweep_id>
```

You can also launch the sweep programmatically:

```bash
python GPU/sweeps/run_graph_sweep.py --entity <your-entity>
```

The sweep explores optimiser settings alongside the Morgan fingerprint radius and bit-length so the data pipeline stays in sync with the model hyperparameters.

To tune the fingerprint models, dedicated sweeps cover each gradient-boosting estimator:

```bash
# CatBoost search across depth, learning-rate, iterations, and regularisation.
wandb sweep GPU/sweeps/fp_catboost_bayesian.yaml

# LightGBM search for tree shape, learning-rate, and sampling ratios.
wandb sweep GPU/sweeps/fp_lightgbm_bayesian.yaml

# XGBoost search over depth, shrinkage, sampling, and regularisation.
wandb sweep GPU/sweeps/fp_xgboost_bayesian.yaml
```

Each configuration keeps the fingerprint radius/bit-length coupled with the estimator-specific hyperparameters so Bayesian optimisation can explore compatible data/feature settings for the selected model (`--model` is fixed by the sweep command).

## TPU workflow

1. Export the PyTorch Geometric dataset to NumPy archives compatible with TF-GNN:

   ```bash
   python TPU/preprocess_to_npz.py --output-dir tf_dataset
   ```

2. Train the TF-GNN model (runs on CPU/GPU by default, pass `--tpu` to target a TPU):

   ```bash
   python TPU/train_tf.py --dataset tf_dataset --model fp_graph --epochs 50 --batch-size 128 --tpu your-tpu-name
   ```

   The trainer now validates that `--batch-size` is a multiple of 64, matching Google’s TPU performance guidelines; 128 is the default for balanced per-core workloads.

   Use `--model` to mirror the PyTorch experiments exactly: `fp` (fingerprint MLP), `graph` (graph-only encoder), `fp_graph` (combined encoder), or `ssiddi`. All models share the same fusion modes, decoder widths, and metric suite as their Lightning counterparts, and additional knobs like `--fusion`, `--final-concat`, `--gnn-layer`, and `--top-k` match the PyTorch configuration options.

3. Run Bayesian optimisation to tune the TensorFlow hyperparameters with W&B sweeps:

   ```bash
   python TPU/tune_tf.py --dataset tf_dataset --model fp_graph --wandb-project your-project --max-trials 40 --epochs 60
   ```

   The CLI launches a W&B Bayesian sweep that samples the encoder width, depth, dropout, activations, attention heads, decoder size, optimiser learning rate, and the fingerprint radius/bit-length. Provide `--raw-data-dir` if you want the tuner to regenerate datasets for unseen fingerprint settings on the fly. Every trial logs metrics, artefacts, and the saved model to W&B; the best run is also exported locally under `tpu_tuning/` by default.

## Docker

Build and run the containerised environment:

```bash
docker build -t ddi-fp-graph .
docker run --gpus all -it --rm \
  -v $(pwd):/workspace ddi-fp-graph \
  --config GPU/configs/graph.yaml
```

The container entrypoint points to `python -m GPU.train`, so any additional CLI flags are appended to the `docker run` command.
