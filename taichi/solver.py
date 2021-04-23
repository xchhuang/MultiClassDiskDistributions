import taichi as ti
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import utils
from time import time

ti.init(ti.cpu)


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
        self.n_rmax = 5
        self.step_size = self.n_rmax / self.nSteps
        self.num_classes = len(categories.keys())
        self.total_samples_per_element = self.samples_per_element + 1

        graph, topological_order = utils.topologicalSort(len(self.categories.keys()), self.relations)
        print('topological_order:', topological_order)

        max_num_disks = -np.inf
        total_num_disks = 0
        for i in categories.keys():
            l = len(categories[i])
            total_num_disks += l
            if max_num_disks < l:
                max_num_disks = l

        # define taichi fields
        # samples_per_element + the center disk
        self.target = ti.field(dtype=ti.f32, shape=(total_num_disks * self.total_samples_per_element * 3))
        self.id2num = ti.field(dtype=ti.i32, shape=self.num_classes)
        self.topological_order = ti.field(dtype=ti.i32, shape=len(topological_order))
        for i in range(len(topological_order)):
            self.topological_order[i] = topological_order[i]
            self.id2num[topological_order[i]] = len(categories[topological_order[i]])

        print(self.topological_order, self.topological_order.shape)
        # print(self.id2num)

        start = time()
        prev_num_elem = 0
        for i in range(self.topological_order.shape[0]):
            num_elem = self.id2num[self.topological_order[i]]
            for j in range(num_elem):
                for k in range(self.total_samples_per_element):
                    for l in range(3):
                        ind = prev_num_elem * self.total_samples_per_element * 3 + (j * self.total_samples_per_element * 3) + (k * 3 + l)
                        # print(ind)
                        print()
                        self.target[ind] = self.categories[topological_order[i]]
            prev_num_elem += num_elem

        end = time()
        print(end - start)

        start = time()
        self.initialize_taichi_fields()
        end = time()
        print(end - start)
        return
        self.output = defaultdict(list)
        self.target_pcf_mean = defaultdict(dict)  # [id][id_parent][nSteps]
        self.target_pcf_min = defaultdict(dict)
        self.target_pcf_max = defaultdict(dict)
        self.target_pcf_radii = defaultdict(dict)
        self.target_pcf_rmax = defaultdict(dict)
        self.target_pcf_area = defaultdict(dict)
        self.target_pcf_count = defaultdict(dict)

        self.output_disks_radii = defaultdict(list)  # [id][num_out_disks_of_idclass], sort radii for synthesis
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
        # self.cur_id = 0
        # self.cur_parent_id = 0
        self.n_repeat = math.ceil(self.n_factor)

        for i in categories.keys():
            x = np.array(categories[i])
            if opt.samples_per_element:
                x = np.expand_dims(x, 1)

            # self.target[i] = ti.field(dtype=ti.f32, shape=x.shape)
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
                self.target_pcf_count[i][j] = Nk
                self.target_pcf_rmax[i][j] = 2 * np.sqrt(1.0 / (2 * np.sqrt(3) * Nk))
                self.target_pcf_radii[i][j] = ti.field(dtype=ti.f32, shape=self.nSteps)
                self.target_pcf_area[i][j] = ti.field(dtype=ti.f32, shape=self.nSteps)

        for i in categories.keys():
            x = np.array(categories[i])
            if opt.samples_per_element:
                x = np.expand_dims(x, 1)
            print('x:', x.shape)  # [N, spe, 3]
            # self.target[i].from_numpy(x)
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
                for k in range(self.nSteps):
                    radii = (i + 1) * self.step_size * self.target_pcf_rmax[i][j]
                    self.target_pcf_radii[i][j][k] = radii
                    inner = max(0, radii - 0.5 * self.target_pcf_rmax[i][j])
                    outer = radii + 0.5 * self.target_pcf_rmax[i][j]
                    self.target_pcf_area[i][j][k] = math.pi * (outer * outer - inner * inner)

        start = time()
        self.compute_target_pcf()
        end = time()
        print('Time for computeTarget {:.4f}s.'.format(end - start))


    @ti.kernel
    def initialize_taichi_fields(self):


    # @ti.kernel
    def compute_target_pcf_kernel(self):
        pass
        # print(self.cur_id, self.cur_parent_id)

    def compute_target_pcf(self):
        for i in self.categories.keys():
            # target_disks = self.target[i]
            for j in self.relations[i]:
                # self.cur_id = i
                # self.cur_parent_id = j
                # print(self.cur_id, self.cur_parent_id)
                self.compute_target_pcf_kernel()
                # cur_pcf_mean = self.target_pcf_model[i][j].pcf_mean.to_numpy()
                # # cur_pcf_mean2 = self.output_pcf_model[i][j].pcf_mean.to_numpy()
                #
                # plt.plot(cur_pcf_mean[:, 0], cur_pcf_mean[:, 1])
                # # print(cur_pcf_mean[:, 1])
                # plt.savefig(
                #     self.opt.output_folder + '/{:}_pcf_{:}_{:}'.format(self.opt.scene_name, i, j))
                # plt.clf()
                # return
