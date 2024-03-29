import json
import os
import os.path as osp

import neptune
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from fp_data import FPDataModule
from models import FPModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger

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


def main():
    # DataModule
    dm = FPDataModule(
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
        "tags": ["Morgan", "concat_first", "full_run"],
        "description": "Morgan (4), full run",
        "name": "Morgan_4",
    }
    model_params = {
        "in_dim": dm.ndim,
        "hid_dim": HID_DIM,
        "out_dim": dm.num_classes,
        "nlayers": NLAYERS,
        "act": "leakyrelu",
        "concat": "first",
        "batch_norm": True,
    }
    early_stopping_params = {
        "monitor": "val_loss",
        "min_delta": 0.0005,
        "patience": 4,
        "verbose": True,
        "mode": "min",
    }
    model_checkpoint_params = {
        "dirpath": osp.join(BASEDIR, "ckpts/", "FP"),
        "filename": "Morgan-{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}",
        "monitor": "val_loss",
        "save_top_k": 1,
    }

    # Logger
    run = neptune.new.init(mode="async", **neptune_params)
    neptune_logger = NeptuneLogger(run=run)

    # Model
    model = FPModel(**model_params)

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
        max_epochs=20,
        progress_bar_refresh_rate=50,
        callbacks=[EarlyStopping(**early_stopping_params), model_checkpoint],
        stochastic_weight_avg=True,
    )

    # Training
    trainer.fit(model, dm)
    trainer.test(model, dm)
    neptune_logger.log_model_summary(model=model, max_depth=-1)
    run.stop()


if __name__ == "__main__":
    main()
