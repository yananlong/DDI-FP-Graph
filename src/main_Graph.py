import json
import os
import os.path as osp

import neptune
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from fp_data import FPGraphDataModule
from models import GraphModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger
from torch import nn

torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PATH"] = (
    "/scratch/midway3/ylong/apps/anaconda/envs/torch_1_10/bin/:" + os.environ["PATH"]
)
pl.seed_everything(2022, workers=True)
BASEDIR = "/project/arzhetsky/ylong/DDI-FP-Graph/"
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
HID_DIM = 256
NLAYERS = 4
GNN = pyg_nn.GINEConv

def main():
    # DataModule
    dm = FPGraphDataModule(
        kind="morgan",
        data_dir=osp.join(BASEDIR, "Data"),
        include_neg=True,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count(),
    )
    dm.setup()

    # Hyperparameters
    neptune_params = {
        "project": "yananlong/DDIFPGraph",
        "tags": ["Morgan", "concat_final", "full_run"],
        "description": "Morgan (4), full run",
        "name": "Morgan_4",
    }
    model_params = {
        "act": "leakyrelu",
        "gnn_name": GNN,
        "gnn_in": 9,  # num input atom features
        "gnn_hid": HID_DIM,
        "dec_hid": HID_DIM,
        "gnn_nlayers": 4,
        "dec_nlayers": 4,
        "out_dim": dm_graph.num_classes,
        "final_concat": True,
        "batch_size": BATCH_SIZE,
    }
    early_stopping_params = {
        "monitor": "val_loss",
        "min_delta": 0.0005,
        "patience": 4,
        "verbose": True,
        "mode": "min",
    }
    model_checkpoint_params = {
        "dirpath": osp.join(BASEDIR, "ckpts/", GNN),
        "filename": "Morgan-{epoch:02d}-{val_loss:.3f}",
        "monitor": "val_loss",
        "save_top_k": 1,
    }

    # Logger
    # run = neptune.new.init(mode="debug", **neptune_params)
    # neptune_logger = NeptuneLogger(run=run)

    # Model
    model = GraphModel(**model_params)

    # Trainer
    model_checkpoint = ModelCheckpoint(**model_checkpoint_params)
    trainer = Trainer(
        # I/O
        default_root_dir=BASEDIR,
        # logger=neptune_logger,
        # Config
        gpus=AVAIL_GPUS,
        auto_select_gpus=True,
        # log_gpu_memory=True,
        # Training
        max_epochs=20,
        progress_bar_refresh_rate=50,
        callbacks=[EarlyStopping(**early_stopping_params), model_checkpoint],
        stochastic_weight_avg=True,
        # profiler="simple",
        # Debugging
        # num_sanity_val_steps=-1,
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
        # limit_test_batches=1,
    )

    # Training
    trainer.fit(model, dm)
    trainer.test(model, dm)
    # neptune_logger.log_model_summary(model=model, max_depth=-1)
    # run.stop()


if __name__ == "__main__":
    main()
