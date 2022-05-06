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
from models import FPModel
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
    dm_morgan = FPDataModule(
        kind="morgan",
        data_dir=osp.join(BASEDIR, "Data"),
        include_neg=True,
        batch_size=BATCH_SIZE,
    )
    dm_morgan.setup()
    early_stopping_params = {
        "monitor": "val_loss",
        "min_delta": 0.0005,
        "patience": 4,
        "verbose": True,
        "mode": "min",
    }
    model_morgan = FPModel(
        act="leakyrelu",
        in_dim=dm_morgan.ndim * 2,
        hid_dim=HID_DIM,
        out_dim=dm_morgan.num_classes,
        nlayers=4,
        dropout=0,
        final_concat=True
    )
    neptune_logger = NeptuneLogger(
        api_key=None,
        project="yananlong/DDIFPGraph",
        tags=["Morgan", "full_run"],
        description="Morgan fingerprint, 4 layers, full run",
    )
    trainer_morgan = Trainer(
        # I/O
        default_root_dir=BASEDIR,
        logger=neptune_logger,
        # Config
        gpus=AVAIL_GPUS,
        auto_select_gpus=True,
        # Training
        max_epochs=40,
        progress_bar_refresh_rate=100,
        callbacks=[EarlyStopping(**early_stopping_params)],
        stochastic_weight_avg=True,
        # Debugging
        # num_sanity_val_steps=-1,
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
        # limit_test_batches=1,
    )
    trainer_morgan.fit(model_morgan, dm_morgan)
    trainer_morgan.test(model_morgan, dm_morgan)
    neptune_logger.log_model_summary(model=model_morgan, max_depth=-1)


if __name__ == "__main__":
    main()
