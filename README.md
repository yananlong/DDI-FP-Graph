# DDI-FP-Graph

Updated training pipelines for the paper [Molecular Fingerprints Are a Simple Yet Effective Solution to the Drug–Drug Interaction Problem](https://icml-compbio.github.io/2022/papers/WCBICML2022_paper_72.pdf).

## What's new?

- **Modern PyTorch Lightning workflows** live under [`PyTorch/`](PyTorch) with W&B integration and Bayesian sweeps, now using the unified `lightning.pytorch` API.
- **TPU-ready TensorFlow GNN pipeline** under [`TPU/`](TPU) for converting the dataset and running on modern TPUs, updated for TensorFlow 2.15 and TF-GNN 1.0.3.
- **Reproducible environments** via the provided [`pyproject.toml`](pyproject.toml) and [`Dockerfile`](Dockerfile).

## Quickstart

Install dependencies with Poetry:

```bash
poetry install
```

Train a graph model with PyTorch Lightning and log to Weights & Biases:

```bash
python -m PyTorch.train --config PyTorch/configs/graph.yaml --run-name dev-run
```

To use the Bayesian sweep configuration:

```bash
wandb sweep PyTorch/sweeps/graph_bayesian.yaml
wandb agent <entity/project>/<sweep_id>
```

You can also launch the sweep programmatically:

```bash
python PyTorch/sweeps/run_graph_sweep.py --entity <your-entity>
```

## TPU workflow

1. Export the PyTorch Geometric dataset to NumPy archives compatible with TF-GNN:

   ```bash
   python TPU/preprocess_to_npz.py --output-dir tf_dataset
   ```

2. Train the TF-GNN model (runs on CPU/GPU by default, pass `--tpu` to target a TPU):

   ```bash
   python TPU/train_tf_gnn.py --dataset tf_dataset --epochs 50 --tpu your-tpu-name
   ```

## Docker

Build and run the containerised environment:

```bash
docker build -t ddi-fp-graph .
docker run --gpus all -it --rm \
  -v $(pwd):/workspace ddi-fp-graph \
  --config PyTorch/configs/graph.yaml
```

The container entrypoint points to `python -m PyTorch.train`, so any additional CLI flags are appended to the `docker run` command.
