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
from asmcdd import ASMCDD
from collections import defaultdict
import utils
from refiner import Refiner


# np.random.seed(0)


class Trainer:
    def __init__(self, device, opt, categories, categories_elem, relations):
        super(Trainer, self).__init__()
        self.device = device
        self.opt = opt

        self.pcf_model = defaultdict(dict)
        self.target_pcfs = defaultdict(dict)

        # if not opt.refine:  # initialization via Metropolis Hastings sampling
        ASMCDD(device, opt, categories, categories_elem, relations)
        # else:  # refine via gradient descent
        #     self.refine(categories, relations)

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
                cur_pcf_model = PCF(self.device, nbbins=self.opt.nSteps, sigma=self.opt.sigma, npoints=len(exemplar_disks),
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
        graph, topological_order = utils.topologicalSort(len(exemplar_categories), exemplar_relations)
        exemplar = []
        output = []
        for k in exemplar_categories.keys():
            exemplar.append(torch.from_numpy(np.array(exemplar_categories[k])).float().to(self.device))
            output.append(torch.from_numpy(np.array(init_categories[k])).float().to(self.device))

        utils.plot_disks(topological_order, exemplar, self.opt.output_folder + '/exemplar')
        utils.plot_disks(topological_order, output, self.opt.output_folder + '/output')

        self.plot_pcf(exemplar, output, exemplar_relations)

        # optimizer = torch.optim.SGD(model_refine.parameters(), lr=0.0001)
        # weight = defaultdict(dict)
        # weight[1][1] = 1
        # weight[1][0] = 50
        # weight[2][2] = 50
        # weight[2][0] = 30

        n_iter = 20

        for i in range(num_classes):
            category_i = i
            exemplar_disks = exemplar[category_i]
            model_refine = Refiner(self.device, self.opt, output[category_i]).to(self.device)
            optimizer = torch.optim.Adam(model_refine.parameters(), lr=1e-3)

            for itr in tqdm(range(n_iter)):
                optimizer.zero_grad()
                out = model_refine()

                output[category_i] = out.clone()

                loss = torch.zeros(1).to(self.device)

                output_disks = output[category_i]

                for j in range(len(exemplar_relations[i])):
                    category_j = exemplar_relations[i][j]
                    output_parent_disks = output[category_j]
                    same_category = False
                    if category_i == category_j:
                        same_category = True
                    cur_pcf_model = self.pcf_model[category_i][category_j]
                    exemplar_pcf_mean = self.target_pcfs[category_i][category_j][:, 0:2].detach()

                    output_pcf_mean = cur_pcf_model(output_disks, output_parent_disks, same_category=same_category,
                                                    dimen=3, use_fnorm=True)

                    cur_loss = torch.nn.functional.mse_loss(output_pcf_mean[:, 1], exemplar_pcf_mean[:, 1])
                    # if category_i == category_j == 0:
                    #     w = 3e-11
                    loss += cur_loss

                loss.backward()
                optimizer.step()
                # if category_i == category_j == 0:
                #     if loss < 2.2e10:
                #         break

                if itr % 10 == 0:
                    print('loss: {:.4f}'.format(loss.item()))
                    self.plot_pcf(exemplar, output, exemplar_relations)
                    utils.plot_disks(topological_order, output, self.opt.output_folder + '/optimized')

            output[category_i] = out.detach()
