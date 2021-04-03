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

# np.random.seed(0)


class ASMCDD(torch.nn.Module):
    def __init__(self, device, opt, categories, relations):
        super(ASMCDD, self).__init__()
        self.device = device
        self.opt = opt
        # print(categories[0])
        # self.w1 = torch.nn.Parameter(torch.from_numpy(np.array(categories[0])).float().to(device))
        # self.w2 = torch.nn.Parameter(torch.from_numpy(np.array(categories[1])).float().to(device))
        # self.w3 = torch.nn.Parameter(torch.from_numpy(np.array(categories[2])).float().to(device))

        self.w = torch.nn.ParameterList()
        for k in categories.keys():
            self.w.append(torch.nn.Parameter(torch.from_numpy(np.array(categories[k])).float()))
        # self.w = torch.nn.ParameterList([self.w1, self.w2, self.w2])

    def forward(self):
        for i in range(3):
            print(self.w[i].shape)
