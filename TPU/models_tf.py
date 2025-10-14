"""TensorFlow models mirroring the PyTorch Lightning implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

import tensorflow as tf
import tensorflow_gnn as tfgnn


_ACTIVATIONS: Dict[str, Callable[[tf.Tensor], tf.Tensor]] = {
    "relu": tf.nn.relu,
    "gelu": tf.nn.gelu,
    "elu": tf.nn.elu,
}


def _activation_fn(name: str | None) -> Callable[[tf.Tensor], tf.Tensor] | None:
    if not name:
        return None
    key = name.lower()
    if key == "leakyrelu":
        return lambda x: tf.nn.leaky_relu(x, alpha=0.2)
    if key in _ACTIVATIONS:
        return _ACTIVATIONS[key]
    return tf.keras.activations.get(name)


def _activation_layer(name: str | None) -> tf.keras.layers.Layer | None:
    if not name:
        return None
    if name.lower() == "leakyrelu":
        return tf.keras.layers.LeakyReLU(alpha=0.2)
    if name.lower() == "prelu":
        return tf.keras.layers.PReLU()
    if name.lower() in _ACTIVATIONS:
        return tf.keras.layers.Activation(_ACTIVATIONS[name.lower()])
    # Fall back to keras built-ins
    return tf.keras.layers.Activation(name)


@dataclass
class FingerprintConfig:
    in_dim: int
    hid_dim: int
    enc_layers: int
    dropout: float
    act: str
    batch_norm: bool = False
    fusion: str = "fingerprint_symmetric"


@dataclass
class GraphConfig:
    gnn_name: str
    gnn_layers: int
    gnn_hidden: int
    atom_in_dim: int
    attn_heads: int
    dropout: float
    act: str
    final_concat: bool


@dataclass
class DecoderConfig:
    hidden_dim: int
    layers: int
    dropout: float
    act: str


@dataclass
class TrainingConfig:
    num_classes: int
    top_k: int = 5


class MacroF1(tf.keras.metrics.Metric):
    """Macro-averaged F1 score."""

    def __init__(self, num_classes: int, name: str = "f1_macro", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=self.num_classes)
        y_pred_one_hot = tf.one_hot(y_pred, depth=self.num_classes)

        tp = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
        fp = tf.reduce_sum((1.0 - y_true_one_hot) * y_pred_one_hot, axis=0)
        fn = tf.reduce_sum(y_true_one_hot * (1.0 - y_pred_one_hot), axis=0)

        if sample_weight is not None:
            sample_weight = tf.reshape(sample_weight, [-1, 1])
            tp *= sample_weight
            fp *= sample_weight
            fn *= sample_weight
            tp = tf.reduce_sum(tp, axis=0)
            fp = tf.reduce_sum(fp, axis=0)
            fn = tf.reduce_sum(fn, axis=0)

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        recall = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        f1 = tf.math.divide_no_nan(2.0 * precision * recall, precision + recall)
        return tf.reduce_mean(f1)

    def reset_states(self):
        for var in (self.tp, self.fp, self.fn):
            var.assign(tf.zeros_like(var))


class WeightedF1(tf.keras.metrics.Metric):
    """Support-weighted F1 score."""

    def __init__(self, num_classes: int, name: str = "f1_weighted", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros")
        self.support = self.add_weight(name="support", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=self.num_classes)
        y_pred_one_hot = tf.one_hot(y_pred, depth=self.num_classes)

        tp = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
        fp = tf.reduce_sum((1.0 - y_true_one_hot) * y_pred_one_hot, axis=0)
        fn = tf.reduce_sum(y_true_one_hot * (1.0 - y_pred_one_hot), axis=0)
        support = tf.reduce_sum(y_true_one_hot, axis=0)

        if sample_weight is not None:
            sample_weight = tf.reshape(sample_weight, [-1, 1])
            tp *= sample_weight
            fp *= sample_weight
            fn *= sample_weight
            support *= sample_weight
            tp = tf.reduce_sum(tp, axis=0)
            fp = tf.reduce_sum(fp, axis=0)
            fn = tf.reduce_sum(fn, axis=0)
            support = tf.reduce_sum(support, axis=0)

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)
        self.support.assign_add(support)

    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        recall = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        f1 = tf.math.divide_no_nan(2.0 * precision * recall, precision + recall)
        weights = tf.math.divide_no_nan(self.support, tf.reduce_sum(self.support))
        return tf.reduce_sum(f1 * weights)

    def reset_states(self):
        for var in (self.tp, self.fp, self.fn, self.support):
            var.assign(tf.zeros_like(var))


class MacroAUROC(tf.keras.metrics.AUC):
    """Macro-averaged AUROC that consumes logits."""

    def __init__(self, num_classes: int, name: str = "auroc_macro", **kwargs):
        super().__init__(
            name=name,
            multi_label=True,
            num_thresholds=200,
            **kwargs,
        )
        self._num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true = tf.one_hot(y_true, depth=self._num_classes)
        probs = tf.nn.softmax(y_pred)
        return super().update_state(y_true, probs, sample_weight)


class CoAttentionLayer(tf.keras.layers.Layer):
    """TensorFlow implementation of the SSI-DDI co-attention module."""

    def __init__(self, features: int, **kwargs):
        super().__init__(**kwargs)
        self.features = features

    def build(self, input_shape):
        half = self.features // 2
        self.w_q = self.add_weight(
            "w_q", shape=(self.features, half), initializer="glorot_uniform"
        )
        self.w_k = self.add_weight(
            "w_k", shape=(self.features, half), initializer="glorot_uniform"
        )
        self.bias = self.add_weight(
            "bias", shape=(half,), initializer="zeros"
        )
        self.a = self.add_weight(
            "a", shape=(half,), initializer="glorot_uniform"
        )

    def call(self, receiver: tf.Tensor, attendant: tf.Tensor) -> tf.Tensor:
        keys = tf.linalg.matmul(receiver, self.w_k)
        queries = tf.linalg.matmul(attendant, self.w_q)
        e_activations = (
            tf.expand_dims(queries, axis=2)
            + tf.expand_dims(keys, axis=1)
            + self.bias
        )
        attentions = tf.tanh(e_activations)
        return tf.tensordot(attentions, self.a, axes=([-1], [0]))


def make_metrics(config: TrainingConfig) -> Iterable[tf.keras.metrics.Metric]:
    return [
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        MacroF1(config.num_classes),
        WeightedF1(config.num_classes),
        MacroAUROC(config.num_classes),
        tf.keras.metrics.TopKCategoricalAccuracy(
            k=config.top_k, name=f"top_{config.top_k}_accuracy"
        ),
    ]


def _apply_fingerprint_fusion(fp1: tf.Tensor, fp2: tf.Tensor, mode: str) -> tf.Tensor:
    alias = {
        "first": "fingerprint_concat",
        "last": "embedding_concat",
        "final": "embedding_sum",
        "sum": "embedding_sum",
    }
    canonical = alias.get(mode.lower(), mode.lower())
    if canonical not in {
        "fingerprint_concat",
        "fingerprint_symmetric",
        "embedding_concat",
        "embedding_sum",
        "embedding_symmetric",
    }:
        raise ValueError(f"Unsupported fusion mode '{mode}'.")
    if canonical == "fingerprint_concat":
        return tf.concat([fp1, fp2], axis=-1)
    if canonical == "fingerprint_symmetric":
        union = fp1 + fp2
        intersection = fp1 * fp2
        exclusive = tf.math.abs(fp1 - fp2)
        return tf.concat([union, intersection, exclusive], axis=-1)
    # The embedding_* modes are handled after encoding.
    return canonical


def _build_mlp(
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    activation: str,
    dropout: float,
    batch_norm: bool = False,
    name: str | None = None,
) -> tf.keras.Sequential:
    layers: list[tf.keras.layers.Layer] = []
    act_layer = _activation_layer(activation)
    for _ in range(max(num_layers - 1, 0)):
        layers.append(tf.keras.layers.Dense(hidden_dim, use_bias=not batch_norm))
        if batch_norm:
            layers.append(tf.keras.layers.BatchNormalization())
        if act_layer is not None:
            layers.append(act_layer.__class__.from_config(act_layer.get_config()))
        if dropout:
            layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(output_dim))
    return tf.keras.Sequential(layers, name=name)


def build_fp_model(
    spec: tfgnn.GraphTensorSpec,
    fingerprint_cfg: FingerprintConfig,
    training_cfg: TrainingConfig,
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(type_spec=spec)
    fp1 = tf.squeeze(inputs.context["fp1"], axis=1)
    fp2 = tf.squeeze(inputs.context["fp2"], axis=1)

    fusion = _apply_fingerprint_fusion(fp1, fp2, fingerprint_cfg.fusion)
    enc_in_dim = fingerprint_cfg.in_dim
    if isinstance(fusion, tf.Tensor):
        enc_inputs = fusion
    else:
        # Embedding modes encode each drug independently.
        encoder = _build_mlp(
            output_dim=fingerprint_cfg.hid_dim,
            hidden_dim=fingerprint_cfg.hid_dim,
            num_layers=fingerprint_cfg.enc_layers,
            activation=fingerprint_cfg.act,
            dropout=fingerprint_cfg.dropout,
            batch_norm=fingerprint_cfg.batch_norm,
            name="fingerprint_encoder",
        )
        d1 = encoder(fp1)
        d2 = encoder(fp2)
        if fusion == "embedding_concat":
            enc_inputs = tf.concat([d1, d2], axis=-1)
        elif fusion == "embedding_sum":
            enc_inputs = d1 + d2
        else:
            enc_inputs = tf.concat([tf.math.abs(d1 - d2), d1 * d2], axis=-1)
        decoder = _build_mlp(
            output_dim=training_cfg.num_classes,
            hidden_dim=fingerprint_cfg.hid_dim,
            num_layers=fingerprint_cfg.enc_layers,
            activation=fingerprint_cfg.act,
            dropout=fingerprint_cfg.dropout,
            batch_norm=fingerprint_cfg.batch_norm,
            name="fingerprint_decoder",
        )
        logits = decoder(enc_inputs)
        return tf.keras.Model(inputs=inputs, outputs=logits)

    # Pre-encoder fusion path (fingerprint_concat / symmetric)
    encoder = _build_mlp(
        output_dim=fingerprint_cfg.hid_dim,
        hidden_dim=fingerprint_cfg.hid_dim,
        num_layers=fingerprint_cfg.enc_layers,
        activation=fingerprint_cfg.act,
        dropout=fingerprint_cfg.dropout,
        batch_norm=fingerprint_cfg.batch_norm,
        name="fingerprint_encoder",
    )
    encoded = encoder(enc_inputs)
    decoder = _build_mlp(
        output_dim=training_cfg.num_classes,
        hidden_dim=fingerprint_cfg.hid_dim,
        num_layers=fingerprint_cfg.enc_layers,
        activation=fingerprint_cfg.act,
        dropout=fingerprint_cfg.dropout,
        batch_norm=fingerprint_cfg.batch_norm,
        name="fingerprint_decoder",
    )
    logits = decoder(encoded)
    return tf.keras.Model(inputs=inputs, outputs=logits)


def _shared_dense(units: int, activation: str, dropout: float) -> tf.keras.Sequential:
    act_fn = _activation_fn(activation)
    layers: list[tf.keras.layers.Layer] = [
        tf.keras.layers.Dense(units, activation=act_fn)
    ]
    if dropout:
        layers.append(tf.keras.layers.Dropout(dropout))
    return tf.keras.Sequential(layers)


def _make_gnn_layer(config: GraphConfig) -> tf.keras.layers.Layer:
    name = config.gnn_name.lower()
    act_fn = _activation_fn(config.act)
    if name in {"gatconv", "gatv2conv"}:
        gat_cls = getattr(tfgnn.keras.layers, "GATv2Conv", None)
        if gat_cls is None:
            raise ValueError("TF-GNN installation does not provide GATv2Conv.")
        if config.gnn_hidden % max(config.attn_heads, 1) != 0:
            raise ValueError(
                "gnn_hidden must be divisible by attn_heads for GAT-style layers."
            )
        per_head = max(config.gnn_hidden // max(config.attn_heads, 1), 1)
        return gat_cls(
            num_heads=config.attn_heads,
            per_head_channels=per_head,
            activation=act_fn,
        )
    return tfgnn.keras.layers.SimpleConv(
        units=config.gnn_hidden,
        activation=act_fn,
        edge_input_feature="bond_feat",
    )


def _apply_gnn_layers(
    inputs: tfgnn.GraphTensor,
    config: GraphConfig,
) -> Tuple[tfgnn.GraphTensor, list[tf.Tensor], list[tf.Tensor]]:
    graph = inputs
    atom_dense = tf.keras.layers.Dense(config.gnn_hidden, name="atom_encoder")
    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn=lambda node_set, features: {
            "atom_feat": atom_dense(features["atom_feat"])
        }
    )(graph)

    pooled_a: list[tf.Tensor] = []
    pooled_b: list[tf.Tensor] = []
    for layer_idx in range(config.gnn_layers):
        conv = _make_gnn_layer(config)
        next_state = _shared_dense(config.gnn_hidden, config.act, config.dropout)
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "drug_a": tfgnn.keras.layers.NodeSetUpdate(
                    edge_sets={"drug_a_bonds": conv},
                    next_state=next_state,
                ),
                "drug_b": tfgnn.keras.layers.NodeSetUpdate(
                    edge_sets={"drug_b_bonds": conv},
                    next_state=next_state,
                ),
            }
        )(graph)
        pooled_a.append(
            tfgnn.keras.layers.Pool(node_set_name="drug_a", reduce_type="mean")(graph)
        )
        pooled_b.append(
            tfgnn.keras.layers.Pool(node_set_name="drug_b", reduce_type="mean")(graph)
        )
    return graph, pooled_a, pooled_b


def build_fp_graph_model(
    spec: tfgnn.GraphTensorSpec,
    fingerprint_cfg: FingerprintConfig,
    graph_cfg: GraphConfig,
    decoder_cfg: DecoderConfig,
    training_cfg: TrainingConfig,
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(type_spec=spec)
    fp1 = tf.squeeze(inputs.context["fp1"], axis=1)
    fp2 = tf.squeeze(inputs.context["fp2"], axis=1)

    fp_encoder = _build_mlp(
        output_dim=fingerprint_cfg.hid_dim,
        hidden_dim=fingerprint_cfg.hid_dim,
        num_layers=fingerprint_cfg.enc_layers,
        activation=fingerprint_cfg.act,
        dropout=fingerprint_cfg.dropout,
        batch_norm=fingerprint_cfg.batch_norm,
        name="fingerprint_encoder",
    )
    fp_embed1 = fp_encoder(fp1)
    fp_embed2 = fp_encoder(fp2)

    _, graph_embeds1, graph_embeds2 = _apply_gnn_layers(inputs, graph_cfg)
    g1 = tf.concat([fp_embed1] + graph_embeds1, axis=-1)
    g2 = tf.concat([fp_embed2] + graph_embeds2, axis=-1)
    if graph_cfg.final_concat:
        fused = tf.concat([g1, g2], axis=-1)
    else:
        fused = g1 + g2

    decoder = _build_mlp(
        output_dim=training_cfg.num_classes,
        hidden_dim=decoder_cfg.hidden_dim,
        num_layers=decoder_cfg.layers,
        activation=decoder_cfg.act,
        dropout=decoder_cfg.dropout,
        name="decoder",
    )
    logits = decoder(fused)
    return tf.keras.Model(inputs=inputs, outputs=logits)


def build_graph_model(
    spec: tfgnn.GraphTensorSpec,
    graph_cfg: GraphConfig,
    decoder_cfg: DecoderConfig,
    training_cfg: TrainingConfig,
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(type_spec=spec)
    _, graph_embeds1, graph_embeds2 = _apply_gnn_layers(inputs, graph_cfg)
    g1 = tf.concat(graph_embeds1, axis=-1)
    g2 = tf.concat(graph_embeds2, axis=-1)
    if graph_cfg.final_concat:
        fused = tf.concat([g1, g2], axis=-1)
    else:
        fused = g1 + g2

    decoder = _build_mlp(
        output_dim=training_cfg.num_classes,
        hidden_dim=decoder_cfg.hidden_dim,
        num_layers=decoder_cfg.layers,
        activation=decoder_cfg.act,
        dropout=decoder_cfg.dropout,
        name="decoder",
    )
    logits = decoder(fused)
    return tf.keras.Model(inputs=inputs, outputs=logits)


def build_ssiddi_model(
    spec: tfgnn.GraphTensorSpec,
    graph_cfg: GraphConfig,
    training_cfg: TrainingConfig,
    att_dim: int,
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(type_spec=spec)

    layer_norm = tf.keras.layers.LayerNormalization()
    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn=lambda node_set, features: {
            "atom_feat": layer_norm(features["atom_feat"])
        }
    )(inputs)

    pooled_repr_a: list[tf.Tensor] = []
    pooled_repr_b: list[tf.Tensor] = []
    if graph_cfg.gnn_hidden % max(graph_cfg.attn_heads, 1) != 0:
        raise ValueError(
            "gnn_hidden must be divisible by attn_heads for SSI-DDI blocks."
        )
    gat_out_dim = graph_cfg.gnn_hidden // max(graph_cfg.attn_heads, 1)
    for _ in range(graph_cfg.gnn_layers):
        gat_cls = getattr(tfgnn.keras.layers, "GATv2Conv", None)
        if gat_cls is None:
            raise ValueError("TF-GNN installation does not provide GATv2Conv.")
        gat = gat_cls(
            num_heads=graph_cfg.attn_heads,
            per_head_channels=gat_out_dim,
            activation="elu",
        )
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "drug_a": tfgnn.keras.layers.NodeSetUpdate(
                    edge_sets={"drug_a_bonds": gat},
                    next_state=tf.keras.layers.LayerNormalization(),
                ),
                "drug_b": tfgnn.keras.layers.NodeSetUpdate(
                    edge_sets={"drug_b_bonds": gat},
                    next_state=tf.keras.layers.LayerNormalization(),
                ),
            }
        )(graph)
        pooled_repr_a.append(
            tfgnn.keras.layers.Pool(node_set_name="drug_a", reduce_type="mean")(graph)
        )
        pooled_repr_b.append(
            tfgnn.keras.layers.Pool(node_set_name="drug_b", reduce_type="mean")(graph)
        )

    stack_a = tf.stack(pooled_repr_a, axis=1)
    stack_b = tf.stack(pooled_repr_b, axis=1)

    co_attention = CoAttentionLayer(att_dim)
    att_weights = co_attention(stack_a, stack_b)
    att_weights = tf.nn.softmax(att_weights, axis=-1)

    stack_a_norm = tf.nn.l2_normalize(stack_a, axis=-1)
    stack_b_norm = tf.nn.l2_normalize(stack_b, axis=-1)
    attended = []
    for i in range(graph_cfg.gnn_layers):
        for j in range(graph_cfg.gnn_layers):
            weight = att_weights[:, i, j][..., tf.newaxis]
            attended.append((stack_a_norm[:, i, :] + stack_b_norm[:, j, :]) * weight)
    fused = tf.concat(attended, axis=-1)

    decoder = _build_mlp(
        output_dim=training_cfg.num_classes,
        hidden_dim=graph_cfg.gnn_hidden,
        num_layers=4,
        activation=graph_cfg.act,
        dropout=graph_cfg.dropout,
        name="decoder",
    )
    logits = decoder(fused)
    return tf.keras.Model(inputs=inputs, outputs=logits)
