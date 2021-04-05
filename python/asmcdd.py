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

# np.random.seed(0)


class ASMCDD(torch.nn.Module):
    def __init__(self, device, opt, categories, relations):
        super(ASMCDD, self).__init__()
        self.device = device
        self.opt = opt
        self.categories = categories
        self.relations = relations

        self.w = torch.nn.ParameterList()
        for k in categories.keys():
            self.w.append(torch.nn.Parameter(torch.from_numpy(np.array(categories[k])).float()))
        self.pcf_model = []
        self.pcf = []
        self.computeTarget()

    def computeTarget(self):
        n_classes = len(self.w)
        for i in range(n_classes):
            category_i = i
            target_disks = self.categories[i]
            for j in range(len(self.relations[i])):
                category_j = self.relations[i][j]
                parent_disks = self.categories[category_j]
                # print(len(target_disk), len(parent_disk))
                cur_pcf_model = PCF(self.device, nbbins=50, sigma=0.25, npoints=len(target_disks), n_rmax=5).to(self.device)
                target_disks = torch.from_numpy(np.array(target_disks)).float().to(self.device)
                parent_disks = torch.from_numpy(np.array(parent_disks)).float().to(self.device)
                same_category = False
                # print(category_i, category_j)
                if category_i == category_j:
                    same_category = True
                cur_pcf_mean, cur_pcf_min, cur_pcf_max = cur_pcf_model(target_disks, parent_disks, same_category=same_category, dimen=3, use_fnorm=True)

                plt.figure(1)
                plt.plot(cur_pcf_mean[:, 0].detach().cpu().numpy(), cur_pcf_mean[:, 1].detach().cpu().numpy())
                plt.savefig(self.opt.output_folder+'/pcf_{:}_{:}'.format(category_i, category_j))
                plt.clf()

                pretty_pcf_model = PrettyPCF(self.device, nbbins=50, sigma=0.25, npoints=len(target_disks), n_rmax=5).to(self.device)
                pretty_pcf = pretty_pcf_model(target_disks, parent_disks, same_category, dimen=3, use_form=False)
                print(pretty_pcf.shape)

    def forward(self):
        for i in range(3):
            print(self.w[i].shape)
