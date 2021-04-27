# import numpy as np
# import matplotlib.pyplot as plt
# import argparse
# from time import time
# import torch
# import math
# from tqdm import tqdm
# import datetime
# import os
# import glob
# from computeFunctions import PCF, PrettyPCF
from asmcdd import ASMCDD
from collections import defaultdict
# import utils
# from refiner import Refiner


class Trainer:
    def __init__(self, device, opt, categories, relations):
        super(Trainer, self).__init__()
        self.device = device
        self.opt = opt

        self.pcf_model = defaultdict(dict)
        self.target_pcfs = defaultdict(dict)

        # if not opt.refine:  # initialization via Metropolis Hastings sampling
        ASMCDD(device, opt, categories, relations)

    def plot_pcf(self, exemplar, output, relations):
        n_classes = len(exemplar)
        for i in range(n_classes):
            category_i = i
            exemplar_disks = exemplar[category_i]
            output_disks = output[category_i]
            for j in range(len(relations[i])):
                plt.figure(1)
                category_j = relations[i][j]
                exemplar_parent_disks = exemplar[category_j]
                output_parent_disks = output[category_j]
                same_category = False
                if category_i == category_j:
                    same_category = True
                cur_pcf_model = PCF(self.device, nbbins=self.opt.nSteps, sigma=self.opt.sigma,
                                    npoints=len(exemplar_disks),
                                    n_rmax=5).to(self.device)
                exemplar_pcf_mean, exemplar_pcf_min, exemplar_pcf_max = cur_pcf_model(exemplar_disks,
                                                                                      exemplar_parent_disks,
                                                                                      same_category=same_category,
                                                                                      dimen=3,
                                                                                      use_fnorm=True)

                output_pcf_mean, output_pcf_min, output_pcf_max = cur_pcf_model(output_disks,
                                                                                output_parent_disks,
                                                                                same_category=same_category,
                                                                                dimen=3,
                                                                                use_fnorm=True)

                self.pcf_model[category_i][category_j] = cur_pcf_model
                self.target_pcfs[category_i][category_j] = torch.cat([exemplar_pcf_mean, exemplar_pcf_min.unsqueeze(1),
                                                                      exemplar_pcf_max.unsqueeze(1)], 1)

                pretty_pcf_model = PrettyPCF(self.device, nbbins=self.opt.nSteps, sigma=self.opt.sigma,
                                             npoints=len(exemplar_disks), n_rmax=5).to(self.device)

                exemplar_pcf_mean = pretty_pcf_model(exemplar_disks, exemplar_parent_disks, same_category, dimen=3,
                                                     use_fnorm=False)
                output_pcf_mean = pretty_pcf_model(output_disks, output_parent_disks, same_category, dimen=3,
                                                   use_fnorm=False)

                self.pcf_model[category_i][category_j] = pretty_pcf_model
                self.target_pcfs[category_i][category_j] = exemplar_pcf_mean

                # plt.figure(1)
                plt.plot(exemplar_pcf_mean[:, 0].detach().cpu().numpy(), exemplar_pcf_mean[:, 1].detach().cpu().numpy(),
                         'r-')
                plt.plot(output_pcf_mean[:, 0].detach().cpu().numpy(), output_pcf_mean[:, 1].detach().cpu().numpy(),
                         'g-')
                plt.legend(['exemplar', 'output'])
                plt.savefig(
                    self.opt.output_folder + '/{:}_pcf_{:}_{:}'.format(self.opt.scene_name, category_i, category_j))
                plt.clf()
        # print('===> Plot PCF Done.')
