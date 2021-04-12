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
from collections import defaultdict
import utils

# np.random.seed(0)


class Trainer:
    def __init__(self, device, opt, categories, relations):
        super(Trainer, self).__init__()
        self.device = device
        self.opt = opt

        if not opt.refine:  # initialization via Metropolis Hastings sampling
            ASMCDD(device, opt, categories, relations)
        else:   # refine via gradient descent
            self.refine(categories, relations)

    def refine(self, exemplar_categories, exemplar_relations):
        init_path = '../simpler_cpp/results_samedomain/' + self.opt.scene_name + '_init.txt'
        init_categories = defaultdict(list)
        with open(init_path) as f:
            id_map = {}  # init
            num_classes = -1  # init
            lines = f.readlines()
            firstline = lines[0].strip().split(' ')
            firstline = [int(x) for x in firstline]
            num_classes = firstline[0]
            class_index = firstline[1:]
            for i in range(num_classes):
                id_map[class_index[i]] = i

            for line in lines[1:]:
                line = line.strip().split(' ')
                line = [int(x) for x in line]
                idx = id_map[line[0]]
                x, y, r = line[1], line[2], line[3]
                # print(idx)
                init_categories[idx].append([x / 10000.0, y / 10000.0, r / 10000.0])

        # print(init_categories)
        # print(exemplar_categories)
        # print(exemplar_relations)
        print(len(exemplar_categories))
        graph, topological_order = utils.topologicalSort(len(exemplar_categories), exemplar_relations)
        print(topological_order)



        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
