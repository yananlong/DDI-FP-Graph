"""Bayesian optimisation utilities for the TensorFlow (TF-GNN) models."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
import tensorflow_gnn as tfgnn

from .models_tf import (
    DecoderConfig,
    FingerprintConfig,
    GraphConfig,
    TrainingConfig,
    build_fp_graph_model,
    build_fp_model,
    build_graph_model,
    build_ssiddi_model,
)
from .train_tf_gnn import (
    compile_model,
    configure_strategy,
    graph_tensor_spec,
    load_metadata,
    load_split,
)


_ACTIVATIONS = ["relu", "leakyrelu", "elu", "gelu"]
_FUSION_MODES = [
    "fingerprint_symmetric",
    "fingerprint_concat",
    "embedding_concat",
    "embedding_sum",
]
_GNN_LAYERS = ["GATConv", "GATv2Conv", "SimpleConv"]


def _build_fp_model_from_config(
    config: dict[str, Any],
    spec: tfgnn.GraphTensorSpec,
    metadata: dict,
    training_cfg: TrainingConfig,
) -> tf.keras.Model:
    cfg = FingerprintConfig(
        in_dim=int(metadata["fp_dim"]),
        hid_dim=int(config["fp_hidden"]),
        enc_layers=int(config["fp_enc_layers"]),
        dropout=float(config["fp_dropout"]),
        act=str(config["fp_activation"]),
        batch_norm=bool(config["fp_batch_norm"]),
        fusion=str(config["fusion"]),
    )
    return build_fp_model(spec, cfg, training_cfg)


def _build_graph_backbone_from_config(
    config: dict[str, Any],
    metadata: dict,
) -> GraphConfig:
    gnn_name = str(config["gnn_layer"])
    gnn_hidden = int(config["gnn_hidden"])
    attn_heads = int(config.get("attn_heads", 1))
    if "gat" in gnn_name.lower() and gnn_hidden % attn_heads != 0:
        attn_heads = 1
    return GraphConfig(
        gnn_name=gnn_name,
        gnn_layers=int(config["gnn_layers"]),
        gnn_hidden=gnn_hidden,
        atom_in_dim=int(metadata["node_dim"]),
        attn_heads=int(attn_heads),
        dropout=float(config["gnn_dropout"]),
        act=str(config["gnn_activation"]),
        final_concat=bool(config["final_concat"]),
    )


def _build_decoder_from_config(config: dict[str, Any]) -> DecoderConfig:
    return DecoderConfig(
        hidden_dim=int(config["dec_hidden"]),
        layers=int(config["dec_layers"]),
        dropout=float(config["dec_dropout"]),
        act=str(config["dec_activation"]),
    )


def _build_model_from_config(
    model_type: str,
    config: dict[str, Any],
    spec: tfgnn.GraphTensorSpec,
    metadata: dict,
    training_cfg: TrainingConfig,
) -> tf.keras.Model:
    if model_type == "fp":
        return _build_fp_model_from_config(config, spec, metadata, training_cfg)

    graph_cfg = _build_graph_backbone_from_config(config, metadata)

    if model_type == "graph":
        dec_cfg = _build_decoder_from_config(config)
        return build_graph_model(spec, graph_cfg, dec_cfg, training_cfg)

    if model_type == "fp_graph":
        fp_cfg = FingerprintConfig(
            in_dim=int(metadata["fp_dim"]),
            hid_dim=int(config["fp_hidden"]),
            enc_layers=int(config["fp_enc_layers"]),
            dropout=float(config["fp_dropout"]),
            act=str(config["fp_activation"]),
            batch_norm=bool(config["fp_batch_norm"]),
            fusion="embedding_concat",
        )
        dec_cfg = _build_decoder_from_config(config)
        return build_fp_graph_model(spec, fp_cfg, graph_cfg, dec_cfg, training_cfg)

    if model_type == "ssiddi":
        att_dim = int(config["ssiddi_att_dim"])
        return build_ssiddi_model(spec, graph_cfg, training_cfg, att_dim=att_dim)

    raise ValueError(f"Unsupported model type '{model_type}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the exported dataset directory.",
    )
    parser.add_argument(
        "--model",
        choices=["fp", "graph", "fp_graph", "ssiddi"],
        default="fp_graph",
        help="Model family to optimise.",
    )
    parser.add_argument(
        "--tpu",
        type=str,
        default=None,
        help="TPU name or address. Leave empty for CPU/GPU.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-trials", type=int, default=20)
    parser.add_argument("--executions-per-trial", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--project-name",
        type=str,
        default="tf_gnn_bayesian_opt",
        help="Name for the W&B sweep group.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="ddi-fp-graph-tpu",
        help="Weights & Biases project where runs will be logged.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional W&B entity/organisation name.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        help="Optional W&B mode override (e.g. 'offline').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tpu_tuning"),
        help="Directory where the best model and reports are exported.",
    )
    return parser.parse_args()


def _sweep_parameters() -> dict[str, Any]:
    return {
        "learning_rate": {"distribution": "log_uniform", "min": 1e-5, "max": 1e-2},
        "fp_hidden": {"distribution": "int_uniform", "min": 128, "max": 512, "step": 64},
        "fp_enc_layers": {"distribution": "int_uniform", "min": 2, "max": 6},
        "fp_dropout": {"distribution": "q_uniform", "min": 0.0, "max": 0.6, "q": 0.1},
        "fp_activation": {"values": _ACTIVATIONS},
        "fp_batch_norm": {"values": [True, False]},
        "fusion": {"values": _FUSION_MODES},
        "gnn_layer": {"values": _GNN_LAYERS},
        "gnn_hidden": {"distribution": "int_uniform", "min": 128, "max": 512, "step": 64},
        "gnn_layers": {"distribution": "int_uniform", "min": 2, "max": 6},
        "gnn_dropout": {"distribution": "q_uniform", "min": 0.0, "max": 0.6, "q": 0.1},
        "gnn_activation": {"values": _ACTIVATIONS},
        "attn_heads": {"values": [1, 2, 4, 8]},
        "final_concat": {"values": [True, False]},
        "dec_hidden": {"distribution": "int_uniform", "min": 128, "max": 512, "step": 64},
        "dec_layers": {"distribution": "int_uniform", "min": 2, "max": 6},
        "dec_dropout": {"distribution": "q_uniform", "min": 0.0, "max": 0.6, "q": 0.1},
        "dec_activation": {"values": _ACTIVATIONS},
        "ssiddi_att_dim": {"distribution": "int_uniform", "min": 128, "max": 512, "step": 64},
    }


def main() -> None:
    args = parse_args()

    if args.batch_size % 64 != 0:
        raise ValueError(
            "TPU batch size must be a multiple of 64 to satisfy per-core alignment recommendations."
        )

    if args.executions_per_trial != 1:
        raise ValueError(
            "Multiple executions per trial are not supported when running W&B sweeps."
        )

    metadata = load_metadata(args.dataset / "metadata.json")
    spec = graph_tensor_spec(metadata)

    train_ds = load_split(args.dataset / "train", spec, shuffle=True, batch_size=args.batch_size)
    val_ds = load_split(args.dataset / "val", spec, shuffle=False, batch_size=args.batch_size)
    test_ds = load_split(args.dataset / "test", spec, shuffle=False, batch_size=args.batch_size)

    training_cfg = TrainingConfig(
        num_classes=int(metadata["num_classes"]),
        top_k=args.top_k,
    )
    strategy = configure_strategy(args.tpu)

    sweep_config = {
        "name": args.project_name,
        "method": "bayes",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": _sweep_parameters(),
    }

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=args.wandb_project,
        entity=args.wandb_entity,
    )

    args.output.mkdir(parents=True, exist_ok=True)

    best_state: dict[str, Any] = {"val_loss": float("inf"), "config": None, "metrics": None}

    def _log_and_maybe_update_best(
        *,
        val_loss: float,
        config: dict[str, Any],
        metrics: dict[str, float],
        model: tf.keras.Model,
    ) -> None:
        if val_loss >= best_state["val_loss"]:
            return

        best_state["val_loss"] = val_loss
        best_state["config"] = config
        best_state["metrics"] = metrics

        model_dir = args.output / "model"
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model.save(str(model_dir))

        with (args.output / "best_hyperparameters.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

        with (args.output / "evaluation.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    def _train_sweep_run() -> None:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.project_name,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "model": args.model,
                "top_k": args.top_k,
            },
        )
        assert run is not None

        sweep_cfg = dict(run.config)

        learning_rate = float(sweep_cfg["learning_rate"])
        with strategy.scope():
            model = _build_model_from_config(
                args.model,
                {k: sweep_cfg[k] for k in _sweep_parameters().keys() if k in sweep_cfg},
                spec,
                metadata,
                training_cfg,
            )
            compile_model(model, learning_rate, training_cfg)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            WandbCallback(save_model=False, log_weights=False),
        ]

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=2,
        )

        val_metrics = model.evaluate(val_ds, return_dict=True, verbose=0)
        test_metrics = model.evaluate(test_ds, return_dict=True, verbose=0)

        wandb.log({f"val/{k}": float(v) for k, v in val_metrics.items()}, commit=False)
        wandb.log({f"test/{k}": float(v) for k, v in test_metrics.items()})

        val_loss = float(val_metrics.get("loss", list(val_metrics.values())[0]))
        run.summary["val_loss"] = val_loss
        for key, value in test_metrics.items():
            run.summary[f"test_{key}"] = float(value)

        run_dir = Path(run.dir)
        hp_path = run_dir / "best_hyperparameters.json"
        with hp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {k: sweep_cfg[k] for k in _sweep_parameters().keys() if k in sweep_cfg},
                handle,
                indent=2,
            )

        eval_path = run_dir / "evaluation.json"
        with eval_path.open("w", encoding="utf-8") as handle:
            json.dump({k: float(v) for k, v in test_metrics.items()}, handle, indent=2)

        model_dir = run_dir / "model"
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model.save(str(model_dir))

        artifact = wandb.Artifact(
            name=f"{run.id}_tf_gnn_model",
            type="model",
            metadata={"val_loss": val_loss},
        )
        artifact.add_file(str(hp_path))
        artifact.add_file(str(eval_path))
        artifact.add_dir(str(model_dir))
        run.log_artifact(artifact)

        combined_metrics = {"val_loss": val_loss}
        combined_metrics.update({f"test_{k}": float(v) for k, v in test_metrics.items()})

        _log_and_maybe_update_best(
            val_loss=val_loss,
            config={k: sweep_cfg[k] for k in _sweep_parameters().keys() if k in sweep_cfg},
            metrics=combined_metrics,
            model=model,
        )

        run.finish()

    wandb.agent(
        sweep_id,
        function=_train_sweep_run,
        count=args.max_trials,
        project=args.wandb_project,
        entity=args.wandb_entity,
    )


if __name__ == "__main__":
    main()

