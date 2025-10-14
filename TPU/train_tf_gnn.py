"""Train a TF-GNN model on TPU using the exported NumPy archives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def graph_tensor_spec(metadata: dict) -> tfgnn.GraphTensorSpec:
    fp_dim = int(metadata["fp_dim"])
    node_dim = int(metadata["node_dim"])
    edge_dim = int(metadata["edge_dim"])
    return tfgnn.GraphTensorSpec.from_piece_specs(
        context_spec=tfgnn.ContextSpec.from_fields(
            features_spec={
                "label": tf.TensorSpec([1], tf.int32),
                "fp1": tf.TensorSpec([1, fp_dim], tf.float32),
                "fp2": tf.TensorSpec([1, fp_dim], tf.float32),
            }
        ),
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


def _npz_to_graph_tensor(npz: dict) -> tfgnn.GraphTensor:
    context = tfgnn.Context.from_fields(
        features={
            "label": tf.convert_to_tensor([npz["label"]], dtype=tf.int32),
            "fp1": tf.convert_to_tensor(npz["fp1"], dtype=tf.float32),
            "fp2": tf.convert_to_tensor(npz["fp2"], dtype=tf.float32),
        }
    )

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

    return tfgnn.GraphTensor.from_pieces(
        context=context,
        node_sets={"drug_a": _node_set("x1"), "drug_b": _node_set("x2")},
        edge_sets={
            "drug_a_bonds": _edge_set("1", "drug_a"),
            "drug_b_bonds": _edge_set("2", "drug_b"),
        },
    )


def load_split(dataset_dir: Path, spec: tfgnn.GraphTensorSpec, shuffle: bool, batch_size: int) -> tf.data.Dataset:
    files = sorted(dataset_dir.glob("*.npz"))

    def generator() -> Iterable[tfgnn.GraphTensor]:
        for path in files:
            with np.load(path) as arrays:
                yield _npz_to_graph_tensor(arrays)

    ds = tf.data.Dataset.from_generator(generator, output_signature=spec)
    if shuffle:
        ds = ds.shuffle(len(files))
    ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model(spec: tfgnn.GraphTensorSpec, metadata: dict, hidden_dim: int, dropout: float) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(type_spec=spec)
    graph = inputs

    def node_update(edge_set_name: str) -> tfgnn.keras.layers.NodeSetUpdate:
        return tfgnn.keras.layers.NodeSetUpdate(
            edge_sets={edge_set_name: tfgnn.keras.layers.SimpleConv(units=hidden_dim, activation="relu")},
            next_state=tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(hidden_dim, activation="relu"),
                    tf.keras.layers.Dropout(dropout),
                ]
            ),
        )

    graph = tfgnn.keras.layers.GraphUpdate(
        node_sets={
            "drug_a": node_update("drug_a_bonds"),
            "drug_b": node_update("drug_b_bonds"),
        }
    )(graph)

    drug_a_state = tfgnn.keras.layers.Pool(node_set_name="drug_a", reduce_type="mean")(graph)
    drug_b_state = tfgnn.keras.layers.Pool(node_set_name="drug_b", reduce_type="mean")(graph)

    pooled_diff = tf.math.abs(drug_a_state - drug_b_state)
    pooled_prod = drug_a_state * drug_b_state
    embedding_features = tf.keras.layers.Concatenate()([pooled_diff, pooled_prod])

    fp1 = tf.squeeze(graph.context["fp1"], axis=1)
    fp2 = tf.squeeze(graph.context["fp2"], axis=1)
    fp_union = fp1 + fp2
    fp_intersection = fp1 * fp2
    fp_exclusive = tf.math.abs(fp1 - fp2)
    fp_features = tf.keras.layers.Concatenate()([fp_union, fp_intersection, fp_exclusive])

    x = tf.keras.layers.Concatenate()([embedding_features, fp_features])
    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    logits = tf.keras.layers.Dense(int(metadata["num_classes"]))(x)

    return tf.keras.Model(inputs=inputs, outputs=logits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the exported dataset directory.")
    parser.add_argument("--tpu", type=str, default=None, help="TPU name or address. Leave empty for CPU/GPU.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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
    spec = graph_tensor_spec(metadata)

    train_ds = load_split(args.dataset / "train", spec, shuffle=True, batch_size=args.batch_size)
    val_ds = load_split(args.dataset / "val", spec, shuffle=False, batch_size=args.batch_size)
    test_ds = load_split(args.dataset / "test", spec, shuffle=False, batch_size=args.batch_size)

    strategy = configure_strategy(args.tpu)
    with strategy.scope():
        model = build_model(spec, metadata, hidden_dim=args.hidden_dim, dropout=args.dropout)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

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
