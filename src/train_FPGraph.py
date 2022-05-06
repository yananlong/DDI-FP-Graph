import json
import os
import os.path as osp

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from fp_data import FPDataModule, FPGraphDataModule
from models import FPGraphModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.utilities import cli
from torch import nn

# https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
torch.multiprocessing.set_sharing_strategy("file_system")

torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

BASEDIR = "/home/yananlong/DDI/"
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
HID_DIM = 256


def main():
    dm_graph_morgan = FPGraphDataModule(
        kind="morgan",
        root=osp.join(BASEDIR, "Data"),
        data_dir=osp.join(BASEDIR, "Data"),
        include_neg=True,
        batch_size=BATCH_SIZE,
    )
    dm_graph_morgan.setup()
    early_stopping_params = {
        "monitor": "val_loss",
        "min_delta": 0.0005,
        "patience": 4,
        "verbose": True,
        "mode": "min",
    }
    model_args = {
        "act": "leakyrelu",
        "fp_hid": HID_DIM,
        "gnn_in": 9,  # num input atom features
        "gnn_hid": HID_DIM,
        "dec_hid": HID_DIM,
        "fp_nlayers": 4,
        "gnn_nlayers": 1,
        "dec_nlayers": 4,
        "batch_size": BATCH_SIZE,
    }
    model_graph_morgan = FPGraphModel(
        fp_in=dm_graph_morgan.ndim,
        gnn_name=pyg_nn.GINEConv,
        out_dim=dm_graph_morgan.num_classes,
        **model_args
    )
    neptune_logger = NeptuneLogger(
        api_key=None,
        project="yananlong/DDIFPGraph",
        tags=["Morgan", "GINEConv", "full_run"],
        description="Morgan fingerprint and GINEConv (1), full run: max 60 epochs",
    )
    trainer_graph_morgan = Trainer(
        # I/O
        default_root_dir=BASEDIR,
        logger=neptune_logger,
        # Config
        gpus=AVAIL_GPUS,
        auto_select_gpus=True,
        # Training
        max_epochs=60,
        progress_bar_refresh_rate=100,
        callbacks=[EarlyStopping(**early_stopping_params)],
        stochastic_weight_avg=True,
        # Debugging
        # num_sanity_val_steps=-1,
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
        # limit_test_batches=1,
    )
    trainer_graph_morgan.fit(model_graph_morgan, dm_graph_morgan)
    trainer_graph_morgan.test(model_graph_morgan, dm_graph_morgan)
    neptune_logger.log_model_summary(model=model_graph_morgan, max_depth=-1)


if __name__ == "__main__":
    main()
