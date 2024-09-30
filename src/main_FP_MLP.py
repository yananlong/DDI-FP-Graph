import json
import os
import os.path as osp

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing
import wandb
import yaml
from fp_data import FPDataModule
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from models import FPModel

torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

BASEDIR = "../"
WANDB_PROJECT = "tabular_mol"

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
pl.seed_everything(2024, workers=True)
BASEDIR = "."
AVAIL_GPUS = torch.cuda.device_count()
NGPUS = 1
# BATCH_SIZE = 256 if AVAIL_GPUS else 64
# HID_DIM = 256
# NLAYERS = 4
# # DROPOUT = 0.5
# RADIUS = 2
# NBITS = 2048
MODE = "inductive1"
TRAIN_PROP = 0.8
VAL_PROP = 0.5


def train():
    # DataModule
    dm = FPDataModule(
        kind="morgan",
        data_dir=osp.join(BASEDIR, "Data"),
        include_neg=True,
        mode=MODE,
        train_prop=TRAIN_PROP,
        val_prop=VAL_PROP,
        batch_size=BATCH_SIZE,
        num_workers=1,
        radius=RADIUS,
        nBits=NBITS,
    )
    dm.setup()

    Hyperparameters
    wandb_params = {
        "project": "yananlong/tabular_mol",
        "tags": ["Morgan", "concat_first", "full_run", MODE],
        "name": f"Morgan{radius}-{nbits}_MLP-{num_layers:d}",
        "notes": f"""
            Morgan FP: radius = {radius}, nbits = {nbits};
            MLP: {num_layers:d} layer(s)
            """,
    }
    model_params = {
        "in_dim": dm.ndim,
        "hid_dim": HID_DIM,
        "out_dim": dm.num_classes,
        "nlayers": NLAYERS,
        # "dropout": DROPOUT,
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

    # Model
    model = FPModel(**model_params)

    # Logger
    wandb_logger = WandbLogger(log_model="all")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")
    early_stopping_callback = EarlyStopping(**early_stopping_params)

    # Trainer
    model_checkpoint = ModelCheckpoint(**model_checkpoint_params)
    trainer = Trainer(
        # I/O
        default_root_dir=BASEDIR,
        logger=wandb_logger,
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
    # neptune_logger.log_model_summary(model=model, max_depth=-1)
    run.stop()


if __name__ == "__main__":
    with open("./sweep_FP_MLP.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    run = wandb.init(
        project=WANDB_PROJECT,
        config=config,
        tags=[
            "lightgbm",
            "dart",
            "baseline",
            "binary_outcome",
            "old_features",
            "sleep_activity",
        ],
    )
    train()
