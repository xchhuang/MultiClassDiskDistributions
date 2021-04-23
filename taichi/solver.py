import taichi as ti
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import utils


@ti.data_oriented
class Contribution:
    def __init__(self, nbbins):
        self.nbbins = nbbins
        self.weights = ti.field(dtype=ti.f32, shape=nbbins)
        self.contribution = ti.field(dtype=ti.f32, shape=nbbins)
        self.pcf = ti.field(dtype=ti.f32, shape=nbbins)

    def initialize(self):
        self.weights.from_numpy(np.zeros(self.nbbins))
        self.contribution.from_numpy(np.zeros(self.nbbins))
        self.pcf.from_numpy(np.zeros(self.nbbins))


@ti.data_oriented
class Solver:
    def __init__(self, opt, categories, relations):
        super(Solver, self).__init__()
        self.opt = opt
        self.categories = categories
        self.relations = relations
        self.nSteps = opt.nSteps
        self.sigma = opt.sigma
        self.domainLength = opt.domainLength
        self.samples_per_element = opt.samples_per_element

        max_num_disks = -np.inf
        for i in categories.keys():
            if max_num_disks < len(categories[i]):
                max_num_disks = len(categories[i])

        # define taichi fields
        self.target = defaultdict(list)
        self.output = defaultdict(list)
        self.target_pcf_mean = defaultdict(dict)  # [id][id_parent][nSteps]
        self.target_pcf_min = defaultdict(dict)
        self.target_pcf_max = defaultdict(dict)
        self.output_disks_radii = defaultdict(
            list)  # [id][num_output_disks_of_id_class], sort radii for synthesis usage
        self.current_pcf = defaultdict(list)  # [id_parent][nSteps]
        self.contributions = defaultdict(Contribution)  # [id_parent][Contribution]
        self.weights = defaultdict(list)  # [relation][num_output_disk, nSteps]
        self.count_disks = defaultdict(int)  # [id], return number of disks already added
        self.d_test_ti = ti.field(dtype=ti.f32, shape=3)  # [samplers_per_element, 3]
        self.test_pcf = Contribution(self.nSteps)  # [Contribution]
        self.tmp_weight = ti.field(dtype=ti.f32, shape=self.nSteps)  # tmp

        self.domainLength = opt.domainLength
        self.n_factor = self.domainLength * self.domainLength
        self.diskfact = 1
        self.cur_id = 0
        self.cur_parent_id = 0
        self.n_repeat = math.ceil(self.n_factor)

        for i in categories.keys():
            x = np.array(categories[i])
            if opt.samples_per_element:
                x = np.expand_dims(x, 1)
            self.target[i] = ti.field(dtype=ti.f32, shape=x.shape)
            self.output[i] = ti.field(dtype=ti.f32, shape=(x.shape[0] * self.n_repeat, x.shape[1], x.shape[2]))
            self.output_disks_radii[i] = ti.field(dtype=ti.f32, shape=len(self.categories[i]) * self.n_repeat)
            self.current_pcf[i] = ti.field(dtype=ti.f32, shape=self.nSteps)
            self.contributions[i] = Contribution(self.nSteps)
            # self.weights[i] = ti.Vector.field(max_num_disks, dtype=ti.f32, shape=self.nSteps)
            for j in self.relations[i]:
                Nk = len(self.categories[i])
                same_category = False
                self.target_pcf_mean[i][j] = ti.field(dtype=ti.f32, shape=self.nSteps)
                self.target_pcf_min[i][j] = ti.field(dtype=ti.f32, shape=self.nSteps)
                self.target_pcf_max[i][j] = ti.field(dtype=ti.f32, shape=self.nSteps)

        for i in categories.keys():
            x = np.array(categories[i])
            if opt.samples_per_element:
                x = np.expand_dims(x, 1)
            # print(x.shape)    # [N, spe, 3]
            self.target[i].from_numpy(x)
            self.output[i].from_numpy(np.zeros((x.shape[0] * self.n_repeat, x.shape[1], x.shape[2])))
            radii = x[:, 0, -1].repeat(self.n_repeat) / self.domainLength
            radii = np.sort(radii)[::-1]
            self.output_disks_radii[i].from_numpy(radii)
            self.current_pcf[i].from_numpy(np.zeros(self.nSteps))
            self.contributions[i].initialize()
            # self.weights[i] = ti.Vector.field(max_num_disks, dtype=ti.f32, shape=self.nSteps)
            for j in self.relations[i]:
                Nk = len(self.categories[i])
                same_category = False
                self.target_pcf_mean[i][j].from_numpy(np.zeros(self.nSteps))
                self.target_pcf_min[i][j].from_numpy(np.ones(self.nSteps) * np.inf)
                self.target_pcf_max[i][j].from_numpy(np.ones(self.nSteps) * -np.inf)

    @ti.kernel
    def compute_pcf(self):
        pass

    def computeTarget(self):
        # n_classes = len(self.target)
        # print('relations:', self.relations)
        for i in self.categories.keys():
            # target_disks = self.target[i]
            for j in self.relations[i]:
                self.target_pcf_model[i][j].forward()
                # self.output_pcf_model[i][j].forward()

                cur_pcf_mean = self.target_pcf_model[i][j].pcf_mean.to_numpy()
                # cur_pcf_mean2 = self.output_pcf_model[i][j].pcf_mean.to_numpy()

                plt.plot(cur_pcf_mean[:, 0], cur_pcf_mean[:, 1])
                # print(cur_pcf_mean[:, 1])
                plt.savefig(
                    self.opt.output_folder + '/{:}_pcf_{:}_{:}'.format(self.opt.scene_name, i, j))
                plt.clf()
                return
        print('===> computeTarget Done.')
