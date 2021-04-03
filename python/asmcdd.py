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
