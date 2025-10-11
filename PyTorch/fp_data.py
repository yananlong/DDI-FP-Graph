import argparse
import json
import os.path as osp
import pickle
from collections import defaultdict
from os import cpu_count
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from sklearn import model_selection
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch_geometric.data import Data as PyGData
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from .SMILES import from_smiles


class FPGraphPairData(PyGData):
    def __init__(
        self,
        fp1=None,
        x1=None,
        edge_index1=None,
        edge_attr1=None,
        fp2=None,
        x2=None,
        edge_index2=None,
        edge_attr2=None,
        y=None,
    ):
        super().__init__()
        self.fp1 = fp1
        self.fp2 = fp2
        self.x1 = x1
        self.x2 = x2
        self.edge_attr1 = edge_attr1
        self.edge_attr2 = edge_attr2
        self.edge_index1 = edge_index1
        self.edge_index2 = edge_index2
        self.y = y

    # Cf.
    # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html#pairs-of-graphs
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index1":
            return self.x1.size(0)
        if key == "edge_index2":
            return self.x2.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class FPDataset(Dataset):
    r"""
    Dataset for fingerprint data generated from SMILES.

    Parameters
    ----------
    data_dir (str): Base directory for input data
    kind (str): Fingerprinting method: morgan or pharmacophores
    include_neg (bool): Use negative examples from actual prescriptions in
        addition to the positive examples from curated databases?
    """

    def __init__(
        self,
        data_dir: str,
        kind: str,
        include_neg: bool = True,
        **kwargs,
    ):
        super().__init__()
        # Unpack keyword args
        try:
            self.radius = kwargs["radius"]
        except KeyError:
            self.radius = 2
        try:
            self.nBits = kwargs["nBits"]
        except KeyError:
            self.nBits = 2048

        # Fingerprints
        self.kind = kind
        if self.kind == "morgan":
            self.dict_in = f"morgan{self.radius}-{self.nBits}_dict_drugbank.pkl"
        elif self.kind == "pharmacophores":
            self.dict_in = "pharmacophore_dict.pkl"
        elif self.kind == "topological":
            self.dict_in = "topological_dict_drugbank.pkl"
        else:
            raise ValueError("Unsupported kind of fingerprinting algorithm")

        # Load fingerprints
        with open(osp.join(data_dir, self.dict_in), mode="rb") as f:
            print("Creating dataset, kind:", kind, flush=True)
            if kind == "morgan":
                print("Radius = {}, #bits = {}".format(self.radius, self.nBits))
            self.fp_dict = pickle.load(f)
        self.ndim = next(iter(self.fp_dict.items()))[1].shape[0]  # input dimensions
        self.dbids_with_fps = list(self.fp_dict.keys())

        # DDI pairs
        self.include_neg = include_neg
        self.ddi_in = "ddi_pos_neg_uniq.tsv" if include_neg else "ddi_pairs.tsv"
        df_ddi = (
            pd.read_table(osp.join(data_dir, self.ddi_in))
            .sort_values(by="ID", ascending=True)
            .reset_index(drop=True)
        )
        df_ddi = df_ddi[
            df_ddi["Drug1"].isin(self.dbids_with_fps)
            & df_ddi["Drug2"].isin(self.dbids_with_fps)
        ]
        self.original_df_ddi = df_ddi.copy()
        self.original_ids = pd.unique(df_ddi[["ID"]].values.ravel("K"))
        self.num_classes = self.original_ids.shape[0]

        # Recode ID to [0, num_classes - 1]
        self.new_ids = (
            df_ddi["ID"]
            .apply(lambda val: np.argwhere(self.original_ids == val).item())
            .to_numpy()
        )
        df_ddi["ID"] = self.new_ids
        self.df_ddi = df_ddi.copy()

    def __len__(self):
        return self.df_ddi.shape[0]

    def __getitem__(self, idx):
        row = self.df_ddi.iloc[idx]
        # row = self.df_ddi.loc[idx]
        dbid1, dbid2, ID = row

        fp1 = torch.FloatTensor(self.fp_dict[dbid1])
        fp2 = torch.FloatTensor(self.fp_dict[dbid2])

        return fp1, fp2, ID


class FPDataModule(pl.LightningDataModule):
    r"""
    DataModule for fingerprint data generated from SMILES.

    Parameters
    ----------
    data_dir (str): Base directory for input data
    kind (str): Fingerprinting method: morgan or pharmacophores
    include_neg (bool): Use negative examples from actual prescriptions in
        addition to the positive examples from curated databases?
    mode (str): transductive or inductive training mode?
    train_prop (np.float32): Under the transductive setting, the proportion of training
    examples as a fraction of *all* data points; under inductive settings number of
    drugs seen during train/validation
    val_prop (np.float32): Proportion of validation examples as a fraction of the
        *remaining* data points; see notes below for behaviour under the different
        settings
    batch_size (int): Batch size
    num_workers (int): Number of processes for data loading

    Notes
    -----
    For inductive settings: in contrast to the transductive setting, here we first
    generate the train/validation samples (i.e. those with the seen drugs) which are
    then split according to `val_prop`. The test set is generated based on how many
    drugs to be held unseen in the previous phases: inductive1 = 1 seen + 1 unseen,
    inductive2 = 2 unseen
    """

    def __init__(
        self,
        data_dir: str,
        kind: str = "morgan",
        include_neg: bool = True,
        mode: str = "transductive",
        train_prop: np.float32 = 0.8,
        val_prop: np.float32 = 0.5,
        batch_size: int = 256,
        num_workers: int = cpu_count(),
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.kind = kind
        self.include_neg = include_neg
        self.mode = mode
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fpargs = kwargs

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Create dataset
        self.ds_full = FPDataset(
            self.data_dir, self.kind, self.include_neg, **self.fpargs
        )
        self.num_classes = self.ds_full.num_classes
        self.ndim = self.ds_full.ndim
        self.dim = (self.ndim, self.num_classes)
        self.nsamples = len(self.ds_full)

        # Train/val/test split
        if self.mode == "transductive":
            self.ntrain = np.int32(self.nsamples * self.train_prop)
            self.nval_test = self.nsamples - self.ntrain
            self.nval = np.int32(self.nval_test * self.val_prop)
            self.ntest = self.nval_test - self.nval
            # `random_split` recodes the indices from 0 to n - 1
            # have to use iloc above and convert to iloc for inductive below
            self.train, self.val, self.test = random_split(
                self.ds_full,
                [self.ntrain, self.nval, self.ntest],
            )
        elif self.mode in ["inductive1", "inductive2"]:
            # Train/val sets
            self.all_drug_ids = np.array(
                list(
                    set(self.ds_full.df_ddi["Drug1"].unique()).union(
                        set(self.ds_full.df_ddi["Drug2"].unique())
                    )
                )
            )
            self.seen_drug_ids = np.random.choice(
                a=self.all_drug_ids,
                size=np.int32(len(self.all_drug_ids) * self.train_prop),
                replace=False,
            )
            self.ds_full.df_ddi_seen = self.ds_full.df_ddi[
                np.logical_and(
                    self.ds_full.df_ddi["Drug1"].isin(self.seen_drug_ids),
                    self.ds_full.df_ddi["Drug2"].isin(self.seen_drug_ids),
                )
            ]
            (
                self.ds_full.df_train,
                self.ds_full.df_val,
            ) = model_selection.train_test_split(
                self.ds_full.df_ddi_seen, train_size=(1 - self.val_prop)
            )
            # Need to convert back to iloc index
            self.train = Subset(
                self.ds_full,
                self.ds_full.df_ddi.index.get_indexer(self.ds_full.df_train.index)
            )
            self.val = Subset(
                self.ds_full,
                self.ds_full.df_ddi.index.get_indexer(self.ds_full.df_val.index)
            )
            self.ntrain = len(self.train)
            self.nval = len(self.val)

            # Test set: generate according to the two settings
            if self.mode == "inductive1":
                self.ds_full.df_ddi_ind = self.ds_full.df_ddi[
                    np.logical_xor(
                        self.ds_full.df_ddi["Drug1"].isin(self.seen_drug_ids),
                        self.ds_full.df_ddi["Drug2"].isin(self.seen_drug_ids),
                    )
                ]
            elif self.mode == "inductive2":
                self.ds_full.df_ddi_ind = self.ds_full.df_ddi[
                    np.logical_and(
                        ~self.ds_full.df_ddi["Drug1"].isin(self.seen_drug_ids),
                        ~self.ds_full.df_ddi["Drug2"].isin(self.seen_drug_ids),
                    )
                ]
            # Need to convert back to iloc index
            self.test = Subset(
                self.ds_full,
                self.ds_full.df_ddi.index.get_indexer(self.ds_full.df_ddi_ind.index)
            )
            self.ntest = len(self.test)

        else:
            raise ValueError("Invalid mode.")

        # Done! Print message
        print(
            "Setting: {}.".format(self.mode),
            "Total #samples {}, #train {}, #validation {}, #test {}".format(
                self.nsamples, self.ntrain, self.nval, self.ntest
            ),
            sep=" ",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size if self.batch_size != -1 else len(self.train),
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size if self.batch_size != -1 else len(self.val),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size if self.batch_size != -1 else len(self.test),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size if self.batch_size != -1 else len(self.test),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


class FPGraphDataset(InMemoryDataset):
    r"""
    Dataset for fingerprint and graph data generated from SMILES.

    Parameters
    ----------
    kind (str): Fingerprinting method: morgan or pharmacophores
    data_dir (str): Base directory for input data
    include_neg (bool): Use negative examples from actual prescriptions in
        addition to the positive examples from curated databases?
    """

    def __init__(
        self,
        root: str,
        kind: str,
        data_dir: str,
        include_neg: bool = True,
        transform: bool = None,
        pre_transform: bool = None,
        pre_filter: bool = None,
        **kwargs,
    ):
        self.kind = kind
        self.data_dir = data_dir
        self.include_neg = include_neg
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.ndim = self.data.fp1.shape[1]
        
        # Unpack keyword args
        try:
            self.radius = kwargs["radius"]
        except KeyError:
            self.radius = 2
        try:
            self.nBits = kwargs["nBits"]
        except KeyError:
            self.nBits = 2048
        
        # Refactored from `process`
        # Fingerprints
        if self.kind == "morgan":
            self.dict_in = f"morgan{self.radius}-{self.nBits}_dict_drugbank.pkl"
        elif self.kind == "pharmacophores":
            self.dict_in = "pharmacophore_dict.pkl"
        elif self.kind == "topological":
            self.dict_in = "topological_dict_drugbank.pkl"
        else:
            raise ValueError("Unsupported kind of fingerprinting algorithm")

        # Load fingerprints
        with open(osp.join(self.data_dir, self.dict_in), mode="rb") as f:
            print("Loading fingerprints of kind:", self.kind, flush=True)
            self.fp_dict = pickle.load(f)
        self.dbids_with_fps = list(self.fp_dict.keys())

        # SMILES
        with open(osp.join(self.data_dir, "dbid_smiles.json"), mode="r") as f:
            print("Loading SMILES", flush=True)
            self.smiles_dict = json.load(fp=f)
        dbids_with_smiles = list(self.smiles_dict.keys())

        # Load DDI pairs
        self.ddi_in = "ddi_pos_neg_uniq.tsv" if self.include_neg else "ddi_pairs.tsv"
        df_ddi = (
            pd.read_table(osp.join(self.data_dir, self.ddi_in))
            .sort_values(by="ID", ascending=True)
            .reset_index(drop=True)
        )

        # Filter DDI pairs
        df_ddi = df_ddi[
            df_ddi["Drug1"].isin(self.dbids_with_fps)
            & df_ddi["Drug2"].isin(self.dbids_with_fps)
            & df_ddi["Drug1"].isin(dbids_with_smiles)
            & df_ddi["Drug2"].isin(dbids_with_smiles)
        ]
        self.original_df_ddi = df_ddi.copy()
        self.original_ids = pd.unique(df_ddi[["ID"]].values.ravel("K"))

        # Recode ID to [0, num_classes - 1]
        self.new_ids = (
            df_ddi["ID"]
            .apply(lambda val: np.argwhere(self.original_ids == val).item())
            .to_numpy()
        )
        df_ddi["ID"] = self.new_ids
        self.df_ddi = df_ddi.copy()

    @property
    def processed_dir(self):
        dir_name = f'GraphFP_{self.kind}{"_neg" if self.include_neg else ""}'
        return osp.join(self.root, dir_name)

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @property
    def uniq_IDs(self) -> int:
        return torch.unique(self.data.ID).numel() if self.data.ID is not None else 0

    def _process_row(self, row):
        i, dbid1, dbid2, ID = row
        if i % 10000 == 0:
            print("", i, sep="\t", flush=True)

        # Fingerprints
        fp1 = torch.FloatTensor(self.fp_dict[dbid1]).unsqueeze(0)
        fp2 = torch.FloatTensor(self.fp_dict[dbid2]).unsqueeze(0)

        # SMILES
        smiles1 = self.smiles_dict[dbid1]
        smiles2 = self.smiles_dict[dbid2]

        # Convert SMILES to graphs
        x1, edge_index1, edge_attr1 = from_smiles(smiles=smiles1, with_hydrogen=True)
        x2, edge_index2, edge_attr2 = from_smiles(smiles=smiles2, with_hydrogen=True)

        data = FPGraphPairData(
            fp1,
            x1,
            edge_index1,
            edge_attr1,
            fp2,
            x2,
            edge_index2,
            edge_attr2,
            ID,
        )

        return data

    def process(self):
        # Make Data list
        print("Processing datalist of DDI pairs:", flush=True)
        data_list = [self._process_row(row) for row in self.df_ddi.itertuples()]

        # Collate data list
        print("Collating datalist...", flush=True)
        data, slices = self.collate(data_list)

        # Save processed data
        torch.save((data, slices), self.processed_paths[0])


class FPGraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        data_dir: str,
        kind: str = "morgan",
        include_neg: bool = True,
        mode: str = "transductive",
        train_prop=0.8,
        val_prop=0.5,
        batch_size: int = 256,
        num_workers: int = cpu_count(),
    ):
        super().__init__()
        self.root = root
        self.data_dir = data_dir
        self.kind = kind
        self.include_neg = include_neg
        self.mode = mode
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    # Original
#     def setup(self, stage=None):
#         self.ds_full = FPGraphDataset(
#             root=self.root,
#             kind=self.kind,
#             data_dir=self.data_dir,
#             include_neg=self.include_neg,
#         )
#         try:
#             self.num_classes = max(self.ds_full.num_classes, self.ds_full.uniq_IDs)
#         except AttributeError:
#             self.num_classes = self.ds_full.num_classes
#         self.ndim = self.ds_full.ndim
#         self.dim = (self.ndim, self.num_classes)
#         self.nsamples = len(self.ds_full)

#         # Train/val/test split
#         self.ntrain = np.int32(self.nsamples * self.train_prop)
#         self.nval_test = self.nsamples - self.ntrain
#         self.nval = np.int32(self.nval_test * self.val_prop)
#         self.ntest = self.nval_test - self.nval
#         self.train, self.val, self.test = random_split(
#             self.ds_full,
#             [self.ntrain, self.nval, self.ntest],
#         )
#         print(
#             "Total #samples {}, #train {}, #validation {}, #test {}".format(
#                 self.nsamples, self.ntrain, self.nval, self.ntest
#             )
#         )
       
    # New
    def setup(self, stage=None):
        # Create dataset
        self.ds_full = FPGraphDataset(
            root=self.root,
            kind=self.kind,
            data_dir=self.data_dir,
            include_neg=self.include_neg,
        )
        try:
            self.num_classes = max(self.ds_full.num_classes, self.ds_full.uniq_IDs)
        except AttributeError:
            self.num_classes = self.ds_full.num_classes
        self.ndim = self.ds_full.ndim
        self.dim = (self.ndim, self.num_classes)
        self.nsamples = len(self.ds_full)

        # Train/val/test split
        if self.mode == "transductive":
            self.ntrain = np.int32(self.nsamples * self.train_prop)
            self.nval_test = self.nsamples - self.ntrain
            self.nval = np.int32(self.nval_test * self.val_prop)
            self.ntest = self.nval_test - self.nval
            # `random_split` recodes the indices from 0 to n - 1
            # have to use iloc above and convert to iloc for inductive below
            self.train, self.val, self.test = random_split(
                self.ds_full,
                [self.ntrain, self.nval, self.ntest],
            )
        elif self.mode in ["inductive1", "inductive2"]:
            # Train/val sets
            self.all_drug_ids = np.array(
                list(
                    set(self.ds_full.df_ddi["Drug1"].unique()).union(
                        set(self.ds_full.df_ddi["Drug2"].unique())
                    )
                )
            )
            self.seen_drug_ids = np.random.choice(
                a=self.all_drug_ids,
                size=np.int32(len(self.all_drug_ids) * self.train_prop),
                replace=False,
            )
            self.ds_full.df_ddi_seen = self.ds_full.df_ddi[
                np.logical_and(
                    self.ds_full.df_ddi["Drug1"].isin(self.seen_drug_ids),
                    self.ds_full.df_ddi["Drug2"].isin(self.seen_drug_ids),
                )
            ]
            (
                self.ds_full.df_train,
                self.ds_full.df_val,
            ) = model_selection.train_test_split(
                self.ds_full.df_ddi_seen, train_size=(1 - self.val_prop)
            )
            # Need to convert back to iloc index
            self.train = Subset(
                self.ds_full,
                self.ds_full.df_ddi.index.get_indexer(self.ds_full.df_train.index)
            )
            self.val = Subset(
                self.ds_full,
                self.ds_full.df_ddi.index.get_indexer(self.ds_full.df_val.index)
            )
            self.ntrain = len(self.train)
            self.nval = len(self.val)

            # Test set: generate according to the two settings
            if self.mode == "inductive1":
                self.ds_full.df_ddi_ind = self.ds_full.df_ddi[
                    np.logical_xor(
                        self.ds_full.df_ddi["Drug1"].isin(self.seen_drug_ids),
                        self.ds_full.df_ddi["Drug2"].isin(self.seen_drug_ids),
                    )
                ]
            elif self.mode == "inductive2":
                self.ds_full.df_ddi_ind = self.ds_full.df_ddi[
                    np.logical_and(
                        ~self.ds_full.df_ddi["Drug1"].isin(self.seen_drug_ids),
                        ~self.ds_full.df_ddi["Drug2"].isin(self.seen_drug_ids),
                    )
                ]
            # Need to convert back to iloc index
            self.test = Subset(
                self.ds_full,
                self.ds_full.df_ddi.index.get_indexer(self.ds_full.df_ddi_ind.index)
            )
            self.ntest = len(self.test)

        else:
            raise ValueError("Invalid mode.")

        # Done! Print message
        print(
            "Setting: {}.".format(self.mode),
            "Total #samples {}, #train {}, #validation {}, #test {}".format(
                self.nsamples, self.ntrain, self.nval, self.ntest
            ),
            sep=" ",
        )

    @property
    def follow_batch(self):
        return ["x1", "x2"]

    def train_dataloader(self):
        return PyGDataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            follow_batch=self.follow_batch,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return PyGDataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            follow_batch=self.follow_batch,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return PyGDataLoader(
            self.test,
            batch_size=self.batch_size,
            # batch_size=len(self.test),
            shuffle=False,
            follow_batch=self.follow_batch,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return PyGDataLoader(
            self.test,
            batch_size=self.batch_size,
            # batch_size=len(self.test),
            shuffle=False,
            follow_batch=self.follow_batch,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
