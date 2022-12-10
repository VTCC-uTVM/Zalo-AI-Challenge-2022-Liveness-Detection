import numpy as np
import time
import random

import os
"""
This file contains helper functions for building the model and for loading model parameters.
These helper functions are built to mirror those in the official TensorFlow implementation.
"""

import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo

SEED = 2021

def seed_torch(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = True

def count_parameters(module):
    print('Number of parameters: {}'.format(
        sum([p.data.nelement() for p in module.parameters()])))
 
def scheduler_lr(optimizer, ep):
    if ep <= 20:
        for p in optimizer.param_groups:
            p['lr'] = 1e-3
    else:
      for p in optimizer.param_groups:
            p['lr'] = 1e-4
