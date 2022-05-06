import os
import os.path as osp
import sys

import dataset
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as geom_nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader
from torchmetrics.functional import accuracy, fbeta

torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

DrugBank_DDI = dataset.DDIPathwayDataset(
    root="/project/arzhetsky/ylong/DDI/Data/",
    name="DDI_pathW",
    ddi_file_name="ddi_pairs.csv",
    db2pw_file_name="dbid_pwid.csv",
    use_edge_weight=True,
    use_edge_compartment=False,
)
