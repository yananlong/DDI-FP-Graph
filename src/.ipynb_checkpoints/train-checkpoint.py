import json
import os
import os.path as osp
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import tensorboard
import torch
import torch.multiprocessing
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from fp_data import FPDataModule, FPGraphDataModule
from models import FPGraphModel
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities import cli
from torch import nn

# Global variables
torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main():
    

if __name__ == "__main__":
    pl_cli = cli.LightningCLI(
        FPGraphModel, FPGraphDataModule, seed_everything_default=2022
    )
