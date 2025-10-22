"""Train TensorFlow (TF-GNN) models that mirror the PyTorch implementations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
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
    make_metrics,
)
from .pairicl import (
    PairICLConfig,
    build_pairicl_support_model,
    build_pairicl_zero_shot_model,
)
from .tabicl import TabICLConfig


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def graph_tensor_spec(
    metadata: dict,
    support_size: int | None = None,
) -> tfgnn.GraphTensorSpec:
    fp_dim = int(metadata["fp_dim"])
    node_dim = int(metadata["node_dim"])
    edge_dim = int(metadata["edge_dim"])

    if support_size is None:
        meta_support = metadata.get("support_size")
        if meta_support is not None:
            support_size = int(meta_support)

    if support_size is not None and support_size <= 0:
        support_size = None

    context_features: dict[str, tf.TensorSpec] = {
        "fp1": tf.TensorSpec([1, fp_dim], tf.float32),
        "fp2": tf.TensorSpec([1, fp_dim], tf.float32),
    }

    if support_size is not None:
        support_length = support_size if support_size > 0 else None
        context_features.update(
            {
                "support_fp1": tf.TensorSpec([support_length, fp_dim], tf.float32),
                "support_fp2": tf.TensorSpec([support_length, fp_dim], tf.float32),
                "support_labels": tf.TensorSpec([support_length], tf.int32),
                "support_mask": tf.TensorSpec([support_length], tf.float32),
            }
        )

        graph_support_dim = metadata.get("support_graph_dim")
        if graph_support_dim is not None:
            graph_support_dim = int(graph_support_dim)
            context_features.update(
                {
                    "support_graph_a": tf.TensorSpec([support_length, graph_support_dim], tf.float32),
                    "support_graph_b": tf.TensorSpec([support_length, graph_support_dim], tf.float32),
                }
            )

        support_embedding_dim = metadata.get("support_embedding_dim")
        if support_embedding_dim is not None:
            support_embedding_dim = int(support_embedding_dim)
            context_features["support_embeddings"] = tf.TensorSpec(
                [support_length, support_embedding_dim], tf.float32
            )

    return tfgnn.GraphTensorSpec.from_piece_specs(
        context_spec=tfgnn.ContextSpec.from_fields(features_spec=context_features),
        node_sets_spec={
            "drug_a": tfgnn.NodeSetSpec.from_fields(
                features_spec={"atom_feat": tf.TensorSpec([None, node_dim], tf.float32)},
                sizes_spec=tf.TensorSpec([None], tf.int32),
            ),
            "drug_b": tfgnn.NodeSetSpec.from_fields(
                features_spec={"atom_feat": tf.TensorSpec([None, node_dim], tf.float32)},
                sizes_spec=tf.TensorSpec([None], tf.int32),
            ),
        },
        edge_sets_spec={
            "drug_a_bonds": tfgnn.EdgeSetSpec.from_fields(
                features_spec={"bond_feat": tf.TensorSpec([None, edge_dim], tf.float32)},
                sizes_spec=tf.TensorSpec([None], tf.int32),
                adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets(
                    source=("drug_a", tf.TensorSpec([None], tf.int32)),
                    target=("drug_a", tf.TensorSpec([None], tf.int32)),
                ),
            ),
            "drug_b_bonds": tfgnn.EdgeSetSpec.from_fields(
                features_spec={"bond_feat": tf.TensorSpec([None, edge_dim], tf.float32)},
                sizes_spec=tf.TensorSpec([None], tf.int32),
                adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets(
                    source=("drug_b", tf.TensorSpec([None], tf.int32)),
                    target=("drug_b", tf.TensorSpec([None], tf.int32)),
                ),
            ),
        },
    )


def _npz_to_graph_tensor_and_label(npz: dict) -> Tuple[tfgnn.GraphTensor, np.ndarray]:
    context_features = {
        "fp1": tf.convert_to_tensor(npz["fp1"], dtype=tf.float32),
        "fp2": tf.convert_to_tensor(npz["fp2"], dtype=tf.float32),
    }

    if "support_fp1" in npz:
        context_features["support_fp1"] = tf.convert_to_tensor(
            npz["support_fp1"], dtype=tf.float32
        )
    if "support_fp2" in npz:
        context_features["support_fp2"] = tf.convert_to_tensor(
            npz["support_fp2"], dtype=tf.float32
        )
    if "support_labels" in npz:
        context_features["support_labels"] = tf.convert_to_tensor(
            npz["support_labels"], dtype=tf.int32
        )
    if "support_mask" in npz:
        context_features["support_mask"] = tf.convert_to_tensor(
            npz["support_mask"], dtype=tf.float32
        )
    if "support_graph_a" in npz:
        context_features["support_graph_a"] = tf.convert_to_tensor(
            npz["support_graph_a"], dtype=tf.float32
        )
    if "support_graph_b" in npz:
        context_features["support_graph_b"] = tf.convert_to_tensor(
            npz["support_graph_b"], dtype=tf.float32
        )
    if "support_embeddings" in npz:
        context_features["support_embeddings"] = tf.convert_to_tensor(
            npz["support_embeddings"], dtype=tf.float32
        )

    context = tfgnn.Context.from_fields(features=context_features)

    def _node_set(name: str) -> tfgnn.NodeSet:
        features = tf.convert_to_tensor(npz[name], dtype=tf.float32)
        return tfgnn.NodeSet.from_fields(
            features={"atom_feat": features},
            sizes=tf.convert_to_tensor([features.shape[0]], dtype=tf.int32),
        )

    def _edge_set(prefix: str, node_name: str) -> tfgnn.EdgeSet:
        edge_attr = tf.convert_to_tensor(npz[f"edge_attr{prefix}"], dtype=tf.float32)
        edge_index = tf.convert_to_tensor(npz[f"edge_index{prefix}"], dtype=tf.int32)
        return tfgnn.EdgeSet.from_fields(
            features={"bond_feat": edge_attr},
            sizes=tf.convert_to_tensor([edge_attr.shape[0]], dtype=tf.int32),
            adjacency=tfgnn.Adjacency.from_indices(
                source=(node_name, edge_index[0]),
                target=(node_name, edge_index[1]),
            ),
        )

    graph = tfgnn.GraphTensor.from_pieces(
        context=context,
        node_sets={"drug_a": _node_set("x1"), "drug_b": _node_set("x2")},
        edge_sets={
            "drug_a_bonds": _edge_set("1", "drug_a"),
            "drug_b_bonds": _edge_set("2", "drug_b"),
        },
    )
    label = np.array(npz["label"], dtype=np.int32)
    return graph, label


def load_split(
    dataset_dir: Path,
    spec: tfgnn.GraphTensorSpec,
    shuffle: bool,
    batch_size: int,
) -> tf.data.Dataset:
    files = sorted(dataset_dir.glob("*.npz"))

    def generator() -> Iterable[Tuple[tfgnn.GraphTensor, np.ndarray]]:
        for path in files:
            with np.load(path) as arrays:
                yield _npz_to_graph_tensor_and_label(arrays)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(spec, tf.TensorSpec([], tf.int32)),
    )
    if shuffle:
        ds = ds.shuffle(len(files))
    ds = ds.batch(batch_size)

    ds = ds.map(lambda graph, label: (graph, tf.cast(label, tf.int32)))
    return ds.prefetch(tf.data.AUTOTUNE)


def compile_model(model: tf.keras.Model, learning_rate: float, training_cfg: TrainingConfig) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=list(make_metrics(training_cfg)),
    )


def build_selected_model(
    args: argparse.Namespace,
    spec: tfgnn.GraphTensorSpec,
    metadata: dict,
) -> Tuple[tf.keras.Model, TrainingConfig]:
    training_cfg = TrainingConfig(num_classes=int(metadata["num_classes"]), top_k=args.top_k)

    if args.model == "fp":
        fp_cfg = FingerprintConfig(
            in_dim=int(metadata["fp_dim"]),
            hid_dim=args.fp_hidden,
            enc_layers=args.fp_enc_layers,
            dropout=args.fp_dropout,
            act=args.fp_activation,
            batch_norm=args.fp_batch_norm,
            fusion=args.fusion,
        )
        model = build_fp_model(spec, fp_cfg, training_cfg)
        return model, training_cfg

    graph_cfg = GraphConfig(
        gnn_name=args.gnn_layer,
        gnn_layers=args.gnn_layers,
        gnn_hidden=args.gnn_hidden,
        atom_in_dim=int(metadata["node_dim"]),
        attn_heads=args.attn_heads,
        dropout=args.gnn_dropout,
        act=args.gnn_activation,
        final_concat=args.final_concat,
    )

    if args.model in {"pairicl_zero", "pairicl_support"}:
        tabicl_cfg = TabICLConfig(
            fingerprint_dim=int(metadata["fp_dim"]),
            embed_dim=args.pairicl_tabicl_hidden or args.pairicl_hidden,
            row_blocks=args.pairicl_tabicl_layers or args.pairicl_layers,
            activation=args.pairicl_tabicl_activation or args.pairicl_activation,
            dropout=(
                args.pairicl_tabicl_dropout
                if args.pairicl_tabicl_dropout is not None
                else args.pairicl_dropout
            ),
            projection_dim=args.pairicl_tabicl_projection or args.pairicl_projection,
            pretrained_path=args.pairicl_tabicl_weights,
            trainable=args.pairicl_tabicl_trainable,
            normalise=not args.pairicl_tabicl_no_normalize,
        )
        pairicl_cfg = PairICLConfig(
            tabicl=tabicl_cfg,
            hidden_dim=args.pairicl_hidden,
            projection_dim=args.pairicl_projection,
            encoder_layers=args.pairicl_layers,
            activation=args.pairicl_activation,
            dropout=args.pairicl_dropout,
            temperature=args.pairicl_temperature,
            support_blend=max(0.0, min(1.0, args.pairicl_support_blend)),
            use_fingerprints=not args.pairicl_disable_fp,
            use_graph_state=not args.pairicl_disable_graph,
        )
        if not pairicl_cfg.use_fingerprints and not pairicl_cfg.use_graph_state:
            raise ValueError("PairICL requires at least one input modality.")
        if args.model == "pairicl_zero":
            model = build_pairicl_zero_shot_model(spec, graph_cfg, pairicl_cfg, training_cfg)
        else:
            model = build_pairicl_support_model(spec, graph_cfg, pairicl_cfg, training_cfg)
        return model, training_cfg

    if args.model == "graph":
        dec_cfg = DecoderConfig(
            hidden_dim=args.dec_hidden,
            layers=args.dec_layers,
            dropout=args.dec_dropout,
            act=args.dec_activation,
        )
        model = build_graph_model(spec, graph_cfg, dec_cfg, training_cfg)
        return model, training_cfg

    if args.model == "fp_graph":
        fp_cfg = FingerprintConfig(
            in_dim=int(metadata["fp_dim"]),
            hid_dim=args.fp_hidden,
            enc_layers=args.fp_enc_layers,
            dropout=args.fp_dropout,
            act=args.fp_activation,
            batch_norm=args.fp_batch_norm,
            fusion="embedding_concat",
        )
        dec_cfg = DecoderConfig(
            hidden_dim=args.dec_hidden,
            layers=args.dec_layers,
            dropout=args.dec_dropout,
            act=args.dec_activation,
        )
        model = build_fp_graph_model(spec, fp_cfg, graph_cfg, dec_cfg, training_cfg)
        return model, training_cfg

    if args.model == "ssiddi":
        model = build_ssiddi_model(
            spec,
            graph_cfg,
            training_cfg,
            att_dim=args.ssiddi_att_dim,
        )
        return model, training_cfg

    raise ValueError(f"Unsupported model type '{args.model}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the exported dataset directory.")
    parser.add_argument("--tpu", type=str, default=None, help="TPU name or address. Leave empty for CPU/GPU.")
    parser.add_argument(
        "--model",
        choices=[
            "fp",
            "graph",
            "fp_graph",
            "ssiddi",
            "pairicl_zero",
            "pairicl_support",
        ],
        default="fp_graph",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=5)

    # Fingerprint options
    parser.add_argument("--fusion", type=str, default="fingerprint_symmetric")
    parser.add_argument("--fp-enc-layers", type=int, default=4)
    parser.add_argument("--fp-hidden", type=int, default=256)
    parser.add_argument("--fp-dropout", type=float, default=0.2)
    parser.add_argument("--fp-activation", type=str, default="leakyrelu")
    parser.add_argument("--fp-batch-norm", action="store_true")

    # Graph / decoder options
    parser.add_argument("--gnn-layer", type=str, default="GATConv")
    parser.add_argument("--gnn-layers", type=int, default=4)
    parser.add_argument("--gnn-hidden", type=int, default=256)
    parser.add_argument("--gnn-dropout", type=float, default=0.5)
    parser.add_argument("--gnn-activation", type=str, default="leakyrelu")
    parser.add_argument("--attn-heads", type=int, default=4)
    parser.add_argument("--final-concat", action="store_true")

    parser.add_argument("--dec-hidden", type=int, default=256)
    parser.add_argument("--dec-layers", type=int, default=4)
    parser.add_argument("--dec-dropout", type=float, default=0.5)
    parser.add_argument("--dec-activation", type=str, default="leakyrelu")

    # SSI-DDI specific options
    parser.add_argument("--ssiddi-att-dim", type=int, default=256)

    # PairICL specific options
    parser.add_argument("--pairicl-hidden", type=int, default=256)
    parser.add_argument("--pairicl-projection", type=int, default=128)
    parser.add_argument("--pairicl-layers", type=int, default=2)
    parser.add_argument("--pairicl-dropout", type=float, default=0.1)
    parser.add_argument("--pairicl-activation", type=str, default="gelu")
    parser.add_argument("--pairicl-temperature", type=float, default=0.07)
    parser.add_argument("--pairicl-support-blend", type=float, default=0.7)
    parser.add_argument("--pairicl-disable-fp", action="store_true")
    parser.add_argument("--pairicl-disable-graph", action="store_true")
    parser.add_argument(
        "--pairicl-tabicl-hidden",
        type=int,
        default=None,
        help="Hidden dimension of the TabICL encoder. Defaults to --pairicl-hidden.",
    )
    parser.add_argument(
        "--pairicl-tabicl-projection",
        type=int,
        default=None,
        help="Projection dimension produced by the TabICL encoder.",
    )
    parser.add_argument(
        "--pairicl-tabicl-layers",
        type=int,
        default=None,
        help="Number of layers in the TabICL encoder. Defaults to --pairicl-layers.",
    )
    parser.add_argument(
        "--pairicl-tabicl-dropout",
        type=float,
        default=None,
        help="Dropout rate for the TabICL encoder. Defaults to --pairicl-dropout.",
    )
    parser.add_argument(
        "--pairicl-tabicl-activation",
        type=str,
        default=None,
        help="Activation function used in the TabICL encoder.",
    )
    parser.add_argument(
        "--pairicl-tabicl-weights",
        type=str,
        default=None,
        help="Path to pretrained TabICL weights (checkpoint, H5 or SavedModel).",
    )
    parser.add_argument(
        "--pairicl-tabicl-trainable",
        action="store_true",
        help="Fine-tune the TabICL encoder instead of freezing it.",
    )
    parser.add_argument(
        "--pairicl-tabicl-no-normalize",
        action="store_true",
        help="Disable the final layer normalisation inside the TabICL encoder.",
    )
    parser.add_argument(
        "--pairicl-support-size",
        type=int,
        default=None,
        help="Override the expected support set size when using PairICL support mode.",
    )

    parser.add_argument("--output", type=Path, default=Path("tpu_checkpoints"))
    return parser.parse_args()


def configure_strategy(tpu: str | None) -> tf.distribute.Strategy:
    if tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu=tpu)
        return tf.distribute.TPUStrategy(resolver)
    return tf.distribute.get_strategy()


def main() -> None:
    args = parse_args()
    if args.batch_size % 64 != 0:
        raise ValueError(
            "TPU batch size must be a multiple of 64 to satisfy per-core alignment recommendations."
        )
    metadata = load_metadata(args.dataset / "metadata.json")
    support_size_override = args.pairicl_support_size
    support_size = None
    if support_size_override is not None:
        support_size = int(support_size_override)
    elif args.model == "pairicl_support":
        meta_support = metadata.get("support_size")
        if meta_support is not None:
            support_size = int(meta_support)
    spec = graph_tensor_spec(metadata, support_size=support_size)

    train_ds = load_split(args.dataset / "train", spec, shuffle=True, batch_size=args.batch_size)
    val_ds = load_split(args.dataset / "val", spec, shuffle=False, batch_size=args.batch_size)
    test_ds = load_split(args.dataset / "test", spec, shuffle=False, batch_size=args.batch_size)

    strategy = configure_strategy(args.tpu)
    with strategy.scope():
        model, training_cfg = build_selected_model(args, spec, metadata)
        compile_model(model, args.learning_rate, training_cfg)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(args.output / "weights"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    eval_results = model.evaluate(test_ds)

    with (args.output / "history.json").open("w", encoding="utf-8") as handle:
        json.dump({"history": history.history, "evaluation": dict(zip(model.metrics_names, eval_results))}, handle, indent=2)

    print("Evaluation:", dict(zip(model.metrics_names, eval_results)))


if __name__ == "__main__":
    main()
