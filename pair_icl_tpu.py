"""
pair_icl_tpu.py
=================

This module provides a minimal end‑to‑end scaffold for doing **PairICL** on
multiple TPU hosts using PyTorch/XLA.  It is designed to illustrate how to
combine the TabICL row–embedding stage with a simple pairwise combiner and an
in‑context transformer for link/pair classification tasks such as drug–drug
interaction (DDI).  The code is intentionally self contained – it does **not**
depend on the TabICL codebase but instead sketches a compatible encoder and
ICL head.  With a few tweaks (e.g. by replacing the `RowEncoder` and
`ICLTransformer` definitions with actual TabICL implementations), you can
directly plug in pretrained TabICL weights.

Key features
------------

* **Multi‑host TPU support**: the script uses the `torch_xla.distributed`
  primitives to run the same program on each TPU host in a pod.  When run via
  `gcloud tpus tpu-vm ssh --worker=all`, each host will spawn one process per
  local TPU core.  The XLA runtime will automatically handle cross‑host
  synchronization and communication.

* **Zero‑shot and support‑set inference**: you can run the model in zero‑shot
  mode (no support examples) or with a fixed support set of labeled pairs
  provided as a numpy/torch file.  The support embeddings are broadcast to all
  devices.

* **Row/Pair datasets**: simple dataset classes illustrate how to build
  per‑entity embeddings and then form pairwise examples.  For real use you
  should replace the synthetic data loading with your own data loaders.

Usage example
-------------

Prepare the data (off device):

>>> # Assume you have a CSV with drug IDs and 2 048‑dim fingerprints
>>> # and another CSV with (drug_id_a, drug_id_b, label) pairs.
>>> # You can precompute row embeddings once and save them.
>>> python pair_icl_tpu.py --mode preprocess \
...   --rows_csv path/to/drugs.csv --pairs_csv path/to/pairs.csv \
...   --output_embeds /tmp/drug_embeds.pt

Zero‑shot prediction on a TPU pod (e.g. v4‑16 with two hosts):

>>> gcloud alpha compute tpus tpu-vm ssh my‑pod --worker=all --command=
...    "PJRT_DEVICE=TPU python3 pair_icl_tpu.py \
...       --mode predict \
...       --row_embeds /tmp/drug_embeds.pt \
...       --pairs_csv path/to/test_pairs.csv \
...       --zero_shot"

Prediction with support set:

>>> python3 pair_icl_tpu.py --mode predict \
...   --row_embeds /tmp/drug_embeds.pt \
...   --pairs_csv path/to/test_pairs.csv \
...   --support_set /tmp/support_pairs.pt

The script can also perform a tiny amount of supervised fine tuning on the
pair‑ICL transformer (not shown here) by adapting the `train` branch.

Disclaimer
----------
This is **example code** intended for demonstration purposes.  It omits many
production concerns (logging, error handling, mixed precision, dynamic shapes,
dataset sharding, etc.).  For research use you will likely replace the model
definitions with the TabICL row encoder and ICL transformer and plug in
pretrained weights.
"""

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.spmd as xsmp
except ImportError as exc:  # pragma: no cover
    xm = None  # type: ignore
    xmp = None  # type: ignore
    xsmp = None  # type: ignore
    # torch_xla is required for TPU execution.  If it's not installed
    # (e.g. in a CPU‑only environment), the code will still run but on CPU.


@dataclass
class RowExample:
    """Simple container for a single entity (row) and its features.

    Attributes
    ----------
    idx: int
        Unique integer identifier for the entity.
    features: torch.Tensor
        Raw feature vector (e.g. fingerprint).  Shape: (p,).
    """

    idx: int
    features: torch.Tensor


@dataclass
class PairExample:
    """Container for a pair of entity indices and a label.

    Attributes
    ----------
    idx_a: int
        Index of the first entity in the pair.
    idx_b: int
        Index of the second entity in the pair.
    label: int
        The class label for the pair (0..C‑1).
    """

    idx_a: int
    idx_b: int
    label: int


class RowDataset(Dataset):
    """Dataset of row examples.

    This dataset simply wraps a list of `RowExample`s.  During preprocessing
    (when `--mode preprocess`), you can load your raw CSV into this dataset and
    feed it to the row encoder to get embeddings.
    """

    def __init__(self, rows: List[RowExample]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor]:
        ex = self.rows[idx]
        return ex.idx, ex.features


class PairDataset(Dataset):
    """Dataset of pair examples.

    Each item returns the indices of the two entities in the pair and the
    associated label.  The row embeddings are looked up externally using the
    indices to avoid duplicating embeddings in the dataset.
    """

    def __init__(self, pairs: List[PairExample]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[int, int, int]:
        p = self.pairs[idx]
        return p.idx_a, p.idx_b, p.label


class RowEncoder(nn.Module):
    """Simple feed‑forward encoder to map raw features to a dense embedding.

    In practice this should be replaced with the pre‑trained TabICL row
    encoder.  The dimensionalities are configurable so you can load
    compatible weights.
    """

    def __init__(self, in_dim: int, embed_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(n_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim
        self.net = nn.Sequential(*layers, nn.Linear(hidden_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ICLTransformer(nn.Module):
    """Transformer encoder performing in‑context learning over pair tokens.

    The transformer reads a sequence of tokens comprising the support set and
    the query set.  It returns logits for each token.  We only use the
    predictions corresponding to the query tokens.  For simplicity this
    implementation uses PyTorch's built in `nn.TransformerEncoder`.

    Parameters
    ----------
    token_dim : int
        Dimensionality of the pair embedding tokens.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer encoder layers.
    dropout : float
        Dropout probability.
    num_classes : int
        Number of output classes.
    """

    def __init__(self, token_dim: int, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1,
                 num_classes: int = 2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=n_heads,
                                                   dim_feedforward=token_dim * 4,
                                                   dropout=dropout, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(token_dim, num_classes)

    def forward(self, tokens: torch.Tensor, type_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        tokens : torch.Tensor
            A tensor of shape (seq_len, batch, token_dim) containing the
            concatenation of support and query pair embeddings.  The sequence
            dimension must come first to satisfy the transformer input shape.
        type_mask : torch.Tensor
            A boolean mask of shape (seq_len, batch) indicating which tokens are
            queries (True) vs supports (False).  Only query tokens will be
            returned in the output logits.

        Returns
        -------
        logits : torch.Tensor
            A tensor of shape (num_query_tokens, batch, num_classes)
            containing the class scores for each query token.
        """
        # The transformer expects shape (seq_len, batch, d_model).
        encoded = self.transformer(tokens)
        logits = self.classifier(encoded)  # (seq_len, batch, num_classes)
        # Flatten along seq_len*batch to filter query tokens
        seq_len, batch_size, num_classes = logits.shape
        logits_flat = logits.reshape(seq_len * batch_size, num_classes)
        type_mask_flat = type_mask.reshape(seq_len * batch_size)
        # Select only query logits
        query_logits = logits_flat[type_mask_flat]
        # Reshape back to (num_queries, batch, num_classes).
        # Note: the number of query tokens may vary per batch; for simplicity,
        # we return a flat tensor and let the caller handle alignment.
        return query_logits


def build_pair_embedding(e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
    """Construct an order‑invariant pair embedding.

    Given two row embeddings `e1` and `e2` of shape (batch, d), compute the
    pair embedding `[|e1 - e2|, e1 * e2]` of shape (batch, 2*d).

    Parameters
    ----------
    e1, e2 : torch.Tensor
        Row embeddings of shape (batch, d).

    Returns
    -------
    torch.Tensor
        Pair embeddings of shape (batch, 2*d).
    """
    diff = (e1 - e2).abs()
    prod = e1 * e2
    return torch.cat([diff, prod], dim=-1)


def preprocess_rows(args: argparse.Namespace) -> None:
    """Precompute row embeddings and save them to disk.

    This mode loads a CSV file containing entity IDs and raw feature vectors,
    constructs a row dataset, runs the row encoder on CPU (or GPU/TPU if
    available) and saves a mapping from entity ID to embedding tensor.  This
    step is typically performed once; the resulting embeddings can then be
    loaded by all hosts for pair predictions.
    """
    import pandas as pd  # local import to avoid unnecessary dependency if unused
    assert args.rows_csv is not None, "--rows_csv is required in preprocess mode"
    assert args.output_embeds is not None, "--output_embeds is required in preprocess mode"
    df = pd.read_csv(args.rows_csv)
    # Expect the CSV to have columns: id, feature_0, feature_1, ..., feature_{p-1}
    feature_cols = [c for c in df.columns if c != 'id']
    rows = [RowExample(idx=int(row['id']), features=torch.tensor(row[feature_cols].astype(float).values))
            for _, row in df.iterrows()]
    dataset = RowDataset(rows)
    device = torch.device('cpu')
    if xm is not None and xm.xla_device_hw() != 'CPU':
        device = xm.xla_device()
    encoder = RowEncoder(in_dim=len(feature_cols), embed_dim=args.embed_dim).to(device)
    encoder.eval()
    embeddings = {}
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=args.batch_size)
        for ids, feats in loader:
            feats = feats.to(device).float()
            emb = encoder(feats)
            for entity_id, vec in zip(ids.tolist(), emb.cpu()):
                embeddings[entity_id] = vec
    # Save embeddings as a dictionary mapping id -> tensor
    with open(args.output_embeds, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Saved {len(embeddings)} embeddings to {args.output_embeds}")


def load_row_embeddings(path: str) -> dict:
    """Load precomputed row embeddings from disk.

    Returns a dictionary mapping entity id to embedding tensor.
    """
    with open(path, 'rb') as f:
        emb = pickle.load(f)
    return {int(k): torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in emb.items()}


def collate_pairs(batch: List[Tuple[int, int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for PairDataset.

    Converts a list of (idx_a, idx_b, label) into tensors.
    """
    idx_a, idx_b, labels = zip(*batch)
    return torch.tensor(idx_a), torch.tensor(idx_b), torch.tensor(labels)


def predict_worker(rank: int, args: argparse.Namespace) -> None:
    """Prediction entry point for each process.

    This function is invoked by `xmp.spawn` on each TPU core within a host.  In
    multi‑host settings the same script is executed on all hosts.  The
    cross‑replica communication primitives provided by torch_xla will ensure
    that embeddings, support sets and predictions are synchronized across the
    entire pod.
    """
    # Choose the appropriate device (TPU or CPU).
    device = torch.device('cpu')
    if xm is not None and xm.xla_device_hw() != 'CPU':
        device = xm.xla_device()

    # Load row embeddings (shared across all hosts).
    row_embeds = load_row_embeddings(args.row_embeds)

    # Build pair dataset
    import pandas as pd  # defer import
    pairs_df = pd.read_csv(args.pairs_csv)
    pair_examples = [PairExample(int(r['id_a']), int(r['id_b']), int(r['label'])) for _, r in pairs_df.iterrows()]
    pair_dataset = PairDataset(pair_examples)
    # Distributed sampler partitions the dataset across global replicas (hosts * cores)
    world_size = xm.xrt_world_size() if xm is not None else 1
    rank_global = xm.get_ordinal() if xm is not None else 0
    sampler = DistributedSampler(pair_dataset, num_replicas=world_size, rank=rank_global, shuffle=False)
    loader = DataLoader(pair_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_pairs)

    # Instantiate model components
    embed_dim = next(iter(row_embeds.values())).shape[-1]
    token_dim = embed_dim * 2  # because pair embedding concatenates two vectors
    # Placeholder encoder – in zero‑shot mode we do not update it
    row_encoder = nn.Identity()  # row embeddings are precomputed; identity placeholder
    # ICL transformer (use TabICL's in‑context model here if available)
    icl_model = ICLTransformer(token_dim=token_dim, n_heads=args.n_heads,
                               n_layers=args.n_layers, dropout=args.dropout,
                               num_classes=args.num_classes).to(device)
    icl_model.eval()

    # Load support set if provided
    support_tokens: Optional[torch.Tensor] = None
    support_labels: Optional[torch.Tensor] = None
    if args.support_set is not None and os.path.exists(args.support_set):
        with open(args.support_set, 'rb') as f:
            support_data = pickle.load(f)
        # support_data is expected to be a list of (pair_token, label) tuples
        s_tokens, s_labels = zip(*support_data)
        support_tokens = torch.stack([torch.tensor(t) for t in s_tokens]).to(device)
        support_labels = torch.tensor(s_labels).to(device)
        # Broadcast support tokens to all devices
        if xm is not None:
            support_tokens = xm.broadcast(support_tokens, 0)
            support_labels = xm.broadcast(support_labels, 0)

    results = []  # store local predictions
    with torch.no_grad():
        for idx_a, idx_b, labels in loader:
            # Look up precomputed embeddings on CPU then move to device
            e1 = torch.stack([row_embeds[int(i)] for i in idx_a.tolist()]).to(device)
            e2 = torch.stack([row_embeds[int(i)] for i in idx_b.tolist()]).to(device)
            pair_tok = build_pair_embedding(e1, e2)  # shape (batch, 2*embed_dim)
            # Construct sequence tokens: [supports + queries]
            if support_tokens is not None:
                tokens = torch.cat([support_tokens, pair_tok], dim=0)  # (S+Q, d)
                # Build type mask: 0 for supports, 1 for queries
                type_mask = torch.cat([
                    torch.zeros(support_tokens.shape[0], dtype=torch.bool),
                    torch.ones(pair_tok.shape[0], dtype=torch.bool)
                ], dim=0)
            else:
                tokens = pair_tok
                type_mask = torch.ones(pair_tok.shape[0], dtype=torch.bool)
            # Transpose to (seq_len, batch, token_dim)
            tokens = tokens.unsqueeze(1)  # (seq_len, 1, d)
            type_mask = type_mask.unsqueeze(1)
            # Forward through transformer
            logits = icl_model(tokens, type_mask)
            # Softmax to probabilities
            probs = F.softmax(logits, dim=-1)
            # Append predictions (probabilities and true labels) for evaluation
            results.extend([(p.cpu(), l) for p, l in zip(probs, labels)])
            if xm is not None:
                xm.mark_step()

    # Gather results across replicas
    if xm is not None:
        gathered = xm.all_gather(results)
        if rank_global == 0:
            results = [item for sublist in gathered for item in sublist]
        else:
            return  # non‑master hosts return early after gather

    # Only master prints results
    if (xm is None) or (rank_global == 0):
        correct = 0
        total = 0
        for prob, true_label in results:
            pred = prob.argmax().item()
            if pred == int(true_label):
                correct += 1
            total += 1
        acc = correct / max(1, total)
        print(f"Predicted {total} pairs; accuracy={acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="PairICL on TPUs with multi‑host support")
    parser.add_argument('--mode', choices=['preprocess', 'predict'], default='predict',
                        help='Operating mode: preprocess rows or predict pairs.')
    parser.add_argument('--rows_csv', type=str, default=None,
                        help='CSV file with rows (id, feature_0, ... feature_p). Required for preprocess.')
    parser.add_argument('--pairs_csv', type=str, default=None,
                        help='CSV file with pairs (id_a, id_b, label). Required for prediction.')
    parser.add_argument('--output_embeds', type=str, default=None,
                        help='Output path for precomputed row embeddings (pickle).')
    parser.add_argument('--row_embeds', type=str, default=None,
                        help='Path to precomputed row embeddings (pickle) for prediction.')
    parser.add_argument('--support_set', type=str, default=None,
                        help='Optional pickle file containing list of (pair_token, label) for support examples.')
    parser.add_argument('--embed_dim', type=int, default=128, help='Dimension of row embeddings when preprocessing.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for embedding and prediction.')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads in the ICL transformer.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers in the ICL transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability in the ICL transformer.')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes in the pair classification task.')
    parser.add_argument('--zero_shot', action='store_true',
                        help='If set, run without any support examples (same as not specifying support_set).')
    args = parser.parse_args()

    if args.mode == 'preprocess':
        preprocess_rows(args)
        return

    # Prediction mode
    assert args.row_embeds is not None, "--row_embeds is required for prediction mode"
    assert args.pairs_csv is not None, "--pairs_csv is required for prediction mode"
    # If zero_shot flag is set, ignore support_set
    if args.zero_shot:
        args.support_set = None
    # Spawn processes for each local TPU core (or single CPU process)
    nprocs = xm.xrt_world_size() if xm is not None else 1
    if xmp is not None and nprocs > 1:
        xmp.spawn(predict_worker, args=(args,), nprocs=nprocs, start_method='fork')
    else:
        predict_worker(0, args)


if __name__ == '__main__':
    main()