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
from models import FPGraphModel
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
NGPUS = 1
BATCH_SIZE = 256 if AVAIL_GPUS else 64
HID_DIM = 256
NLAYERS = 4
GNN = pyg_nn.GCNConv


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
        "tags": [GNN.__name__, "Morgan", "concat_final", "full_run"],
        "description": "Morgan {} ({}), full run".format(GNN.__name__, NLAYERS),
        "name": "Morgan_{}_{}".format(GNN.__name__, NLAYERS),
        "source_files": ["src/main_FPGraph.py", "src/models.py"],
    }
    model_params = {
        "act": "leakyrelu",
        "fp_nlayers": NLAYERS,
        "fp_in": dm.ndim,
        "fp_hid": HID_DIM,
        "gnn_name": GNN,
        "gnn_nlayers": NLAYERS,
        "gnn_in": 9,  # num input atom features
        "gnn_hid": HID_DIM,
        "dec_nlayers": NLAYERS,
        "dec_hid": HID_DIM,
        "out_dim": dm.num_classes,
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
        "dirpath": osp.join(BASEDIR, "ckpts/", "FPGNN"),
        "filename": "Morgan-{}".format(GNN.__name__) + "-{epoch:02d}-{val_loss:.3f}",
        "monitor": "val_loss",
        "save_top_k": 1,
    }

    # Logger
    run = neptune.new.init(mode="async", **neptune_params)
    neptune_logger = NeptuneLogger(run=run)

    # Model
    model = FPGraphModel(**model_params)

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
