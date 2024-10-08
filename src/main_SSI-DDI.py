import json
import os
import os.path as osp

import neptune
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from fp_data import FPGraphDataModule
from models import SSIDDIModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger

# https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
pl.seed_everything(2023, workers=True)
BASEDIR = "."
AVAIL_GPUS = torch.cuda.device_count()
NGPUS = 1
BATCH_SIZE = 256 if AVAIL_GPUS else 64
HID_DIM = 256
NLAYERS = 5
MODE = "inductive1"
TRAIN_PROP = 0.8
VAL_PROP = 1 / 9


def main():
    # DataModule
    dm = FPGraphDataModule(
        kind="morgan",
        root=osp.join(BASEDIR, "Data"),
        data_dir=osp.join(BASEDIR, "Data"),
        include_neg=True,
        mode=MODE,
        train_prop=TRAIN_PROP,
        val_prop=VAL_PROP,
        batch_size=BATCH_SIZE,
        # There appears to be memory leaks when num_workers > 1
        # cf. https://github.com/pyg-team/pytorch_geometric/issues/3396
        # setting `num_workers` to 1 solves the issue
        num_workers=1,
    )
    dm.setup()

    # Hyperparameters
    model_params = {
        "batch_size": BATCH_SIZE,
        "act": "leakyrelu",
        "in_dim": 9,  # num input atom features
        "hid_dim": HID_DIM,
        "GAT_head_dim": 64,  # 32
        "GAT_nheads": 4,  # 2
        "GAT_nlayers": NLAYERS,  # 4
        "out_dim": dm.num_classes,
    }
    neptune_params = {
        "project": "DDI/fingerprint",
        "tags": ["SSI-DDI", "full_run", MODE],
        "description": "SSI-DDI-v2: {} GAT layers, {} * {}, full run".format(
            NLAYERS, model_params["GAT_head_dim"], model_params["GAT_nheads"]
        ),
        "name": "SSI-DDI-v2_{}_{}_{}".format(
            NLAYERS, model_params["GAT_head_dim"], model_params["GAT_nheads"]
        ),
        "source_files": ["src/main_SSI-DDI.py", "src/models.py", "src/ssiddi.py"],
    }
    early_stopping_params = {
        "monitor": "val_loss",
        "min_delta": 0.0005,
        "patience": 4,
        "verbose": True,
        "mode": "min",
    }
    model_checkpoint_params = {
        "dirpath": osp.join(BASEDIR, "ckpts/", "SSI-DDI"),
        "filename": "SSI-DDI-{epoch:02d}-{val_loss:.3f}",
        "monitor": "val_loss",
        "save_top_k": 1,
    }

    # Logger
    run = neptune.new.init(mode="async", **neptune_params)
    neptune_logger = NeptuneLogger(run=run)
    neptune_logger.experiment["model/hyper-parameters"] = model_params

    # Model
    model = SSIDDIModel(**model_params)

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
    )

    # Training
    trainer.fit(model, dm)
    trainer.test(model, dm)
    neptune_logger.log_model_summary(model=model, max_depth=-1)
    run.stop()


if __name__ == "__main__":
    main()
