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
from computeFunctions import PCF
from asmcdd import ASMCDD

# np.random.seed(0)


class Trainer:
    def __init__(self, device, opt, categories, relations):
        super(Trainer, self).__init__()
        self.device = device
        self.opt = opt
        self.model = ASMCDD(device, opt, categories, relations)
        self.model.to(device)
        self.train()

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # self.model()