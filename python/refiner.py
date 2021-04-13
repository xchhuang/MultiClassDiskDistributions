import numpy as np
import matplotlib.pyplot as plt
import argparse
from time import time
import torch
import math
from tqdm import tqdm
import datetime
import os
import glob
from computeFunctions import PCF, PrettyPCF
from collections import defaultdict
from utils import Contribution
from computeFunctions import compute_contribution, compute_error
import utils
import copy
from graphlib import TopologicalSorter


# np.random.seed(0)


class Refiner(torch.nn.Module):
    def __init__(self, device, opt, output, trainable_index):
        super(Refiner, self).__init__()
        self.device = device
        self.opt = opt
        self.nSteps = opt.nSteps
        self.trainable_index = trainable_index

        self.w_pos = torch.nn.ParameterList()
        self.w_feat = []    # features are fixed, not learnable

        self.w_pos.append(torch.nn.Parameter(output[trainable_index][:, 0:2]))
        self.w_feat.append(output[trainable_index][:, 2:])

    def forward(self):
        pos = torch.clamp(self.w_pos[0], 0, 1)
        out = torch.cat([pos, self.w_feat[0]], 1)
        # print(out.shape)
        return out
