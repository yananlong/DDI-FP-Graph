import json
import os
import os.path as osp

import neptune
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing
import torch_geometric.nn as pyg_nn
from fp_data import FPGraphDataModule
from models import GraphModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger

# https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
pl.seed_everything(2022, workers=True)
BASEDIR = "."
AVAIL_GPUS = torch.cuda.device_count()
NGPUS = 2
BATCH_SIZE = 256 if AVAIL_GPUS else 64
HID_DIM = 256
NLAYERS = 4
GNN = pyg_nn.GINEConv
GNN_NAME = GNN.__name__


def main():
    # DataModule
    dm = FPGraphDataModule(
        kind="morgan",
        root=osp.join(BASEDIR, "Data"),
        data_dir=osp.join(BASEDIR, "Data"),
        include_neg=True,
        batch_size=BATCH_SIZE,
        num_workers=int(os.cpu_count() // AVAIL_GPUS * NGPUS),
    )
    dm.setup()

    # Hyperparameters
    neptune_params = {
        "project": "yananlong/DDIFPGraph",
        "tags": ["Morgan", "concat_final", "full_run"],
        "description": "Morgan (4), full run",
        "name": "Morgan_4",
        "source_files": ["main_Graph.py", "src/models.py"],
    }
    model_params = {
        "act": "leakyrelu",
        "gnn_name": GNN_NAME,
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
        "dirpath": osp.join(BASEDIR, "ckpts/", "GNN"),
        "filename": "{}".format(GNN_NAME) + "-{epoch:02d}-{val_loss:.3f}",
        "monitor": "val_loss",
        "save_top_k": 1,
    }

    # Logger
    run = neptune.new.init(mode="async", **neptune_params)
    neptune_logger = NeptuneLogger(run=run)

    # Model
    model = GraphModel(**model_params)

    # Trainer
    model_checkpoint = ModelCheckpoint(**model_checkpoint_params)
    trainer = Trainer(
        # I/O
        default_root_dir=BASEDIR,
        logger=neptune_logger,
        # Config
        gpus=NGPUS,
        auto_select_gpus=True,
        # Training
        max_epochs=50,
        progress_bar_refresh_rate=50,
        callbacks=[EarlyStopping(**early_stopping_params), model_checkpoint],
        stochastic_weight_avg=True,
        profiler="simple",
    )

    # Training
    trainer.fit(model, dm)
    trainer.test(model, dm)
    neptune_logger.log_model_summary(model=model, max_depth=-1)
    run.stop()


if __name__ == "__main__":
    main()
