import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from graphlib import TopologicalSorter
import taichi as ti


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


def diskDistance(a, b, rmax):
    n_dim = a.size(-1)
    # print(a.shape)
    # print(b.shape)
    radius = torch.cat((a[:, n_dim - 1:n_dim], b[:, n_dim - 1:n_dim]), dim=1)
    # print(radius)
    r1 = torch.max(radius, dim=1)[0] / rmax
    r2 = torch.min(radius, dim=1)[0] / rmax
    # print(r2)
    d = torch.norm(a[:, 0:2] - b[:, 0:2], dim=1) / rmax
    # print(d)
    extent = torch.max(d + r1 + r2, 2 * r1)
    # print(extent)
    overlap = torch.clamp(r1 + r2 - d, 0)
    # print(overlap)
    idx = overlap > 2 * r2
    overlap[idx] = 2 * r2[idx]
    # print(overlap)
    f = extent - overlap + d + r1 - r2
    # print(f)

    idx1 = d <= r1 - r2
    d_norm = torch.zeros_like(d)

    # print (d.size())

    if d_norm[idx1].size(0) > 0:
        d_norm[idx1] = f[idx1] / (4 * r1[idx1] - 4 * r2[idx1])
        # print(d_norm[idx1])

    idx2 = d > r1 - r2
    idx3 = d <= r1 + r2
    # idx23 = idx2 * idx3
    idx23 = torch.logical_and(idx2, idx3)
    # print (d[idx23].size())
    if d_norm[idx23].size(0) > 0:
        d_norm[idx23] = (f[idx23] - 4 * r1[idx23] + 7 * r2[idx23]) / (3 * r2[idx23])
        # print('d <= r1 + r2:', d_norm[idx23])

    idx4 = d > r1 + r2
    # print (d[idx4].size())
    if d_norm[idx4].size(0) > 0:
        d_norm[idx4] = f[idx4] - 4 * r1[idx4] - 2 * r2[idx4] + 3
        # print('else:', d_norm[idx4])

    return d_norm


colors_dict = {
    0: 'r',
    1: 'g',
    2: 'b',
    3: 'k',
}


def plot_disks(topological_order, outputs, filename):
    fig, ax = plt.subplots()
    for k in topological_order:
        # if k > 1:
        #     break
        # print(k)
        out = outputs[k]
        # print(out.shape)
        # out = torch.stack(out, 0).detach().cpu().numpy()
        out = out.detach().cpu().numpy()
        # plt.subplot(121)
        for i in range(out.shape[0]):
            # plt.scatter(out[:, 0], out[:, 1], s=5)
            circle = plt.Circle((out[i, 0], out[i, 1]), out[i, 2], color=colors_dict[k], fill=False)
            ax.add_artist(circle)
        plt.axis('equal')
        # plt.xlim([-0.5, 2.5])
        # plt.ylim([-0.5, 2.5])
        plt.xlim([-0.2, 1.2])
        plt.ylim([-0.2, 1.2])
        # plt.title('Synthesized')
    plt.savefig(filename)
    plt.clf()


def plot_disks_ti(topological_order, outputs, target_id2ElemNum, target_id2PrevElemNum, total_samples_per_element, n_repeat, domainLength, filename):

    # topological_order = topological_order.to_numpy()
    outputs = outputs.to_numpy()
    target_id2ElemNum = target_id2ElemNum.to_numpy()
    target_id2PrevElemNum = target_id2PrevElemNum.to_numpy()

    fig, ax = plt.subplots()
    for i in range(len(topological_order)):
        out = []
        num_elem = target_id2ElemNum[topological_order[i]] * n_repeat
        for j in range(num_elem):
            e = []
            for k in range(total_samples_per_element):
                ind = target_id2PrevElemNum[topological_order[i]] * total_samples_per_element * 3 + (
                        j * total_samples_per_element * 3) + (k * 3)
                e.append([outputs[ind + 0]/domainLength, outputs[ind + 1]/domainLength, outputs[ind + 2]/domainLength])
            out.append(e)
        out = np.array(out)

        # plt.subplot(121)
        for a in range(out.shape[0]):
            for b in range(out.shape[1]):
                # plt.scatter(out[:, 0], out[:, 1], s=5)
                circle = plt.Circle((out[a, b, 0], out[a, b, 1]), out[a, b, 2], linewidth=1,
                                    color=colors_dict[topological_order[i]], fill=False)
                ax.add_artist(circle)
        plt.axis('equal')
        # plt.xlim([-0.5, 2.5])
        # plt.ylim([-0.5, 2.5])
        plt.xlim([-0.2, 1.2])
        plt.ylim([-0.2, 1.2])
        # plt.title('Synthesized')
    plt.savefig(filename)
    plt.clf()


def plot_pcf(topological_order, relations, pcf_mean, pcf_min, pcf_max, target_id2RelNum, target_id2PrevRelNum, nSteps, filename):
    # topological_order = topological_order.to_numpy()
    pcf_mean = pcf_mean.to_numpy()
    target_id2RelNum = target_id2RelNum.to_numpy()
    target_id2PrevRelNum = target_id2PrevRelNum.to_numpy()

    for i in range(len(topological_order)):
        id = topological_order[i]
        num_relation = target_id2RelNum[id]
        for j in range(num_relation):
            out_pcf_mean = []
            out_pcf_min = []
            out_pcf_max = []

            for k in range(nSteps):
                # print(self.target_id2PrevElemNum[self.topological_order[i]])
                ind_w = target_id2PrevRelNum[id] * nSteps + j * nSteps + k
                out_pcf_mean.append(pcf_mean[ind_w])
                out_pcf_min.append(pcf_min[ind_w])
                out_pcf_max.append(pcf_max[ind_w])

            out_pcf_mean = np.array(out_pcf_mean)
            out_pcf_min = np.array(out_pcf_min)
            out_pcf_max = np.array(out_pcf_max)

            # print(out.shape)
            plt.figure(1)
            plt.plot(out_pcf_mean, c='r')
            plt.plot(out_pcf_min, c='g')
            plt.plot(out_pcf_max, c='b')
            plt.savefig(filename+'_{:}_{:}'.format(id, relations[id][j]))
            plt.clf()


def plot_2_pcf(topological_order, relations, target_pcf_mean, output_pcf_mean, target_id2RelNum, target_id2PrevRelNum, nSteps, filename):
    # topological_order = topological_order.to_numpy()
    target_pcf_mean = target_pcf_mean.to_numpy()
    target_id2RelNum = target_id2RelNum.to_numpy()
    target_id2PrevRelNum = target_id2PrevRelNum.to_numpy()

    for i in range(len(topological_order)):
        id = topological_order[i]
        num_relation = target_id2RelNum[id]
        for j in range(num_relation):
            output_pcf = []
            target_pcf = []

            for k in range(nSteps):
                # print(self.target_id2PrevElemNum[self.topological_order[i]])
                ind_w = target_id2PrevRelNum[id] * nSteps + j * nSteps + k
                target_pcf.append(target_pcf_mean[ind_w])
                output_pcf.append(output_pcf_mean[ind_w])

            target_pcf = np.array(target_pcf)
            output_pcf = np.array(output_pcf)

            # print(out.shape)
            plt.figure(1)
            plt.plot(output_pcf, c='g')
            plt.plot(target_pcf, c='r')
            plt.savefig(filename+'_{:}_{:}'.format(id, relations[id][j]))
            plt.clf()


def topologicalSort(num_classes, relations):
    graph = defaultdict(list)
    for k in range(num_classes):
        if len(relations[k]) == 1:
            graph[k] = []
        else:
            for v in relations[k]:
                if v != k:
                    graph[k].append(v)

    ts = TopologicalSorter(graph)
    topological_order = list(ts.static_order())

    return graph, topological_order  # , root_id


def toroidalWrapAround(points, domain_size=1):
    points = torch.where(torch.gt(points, domain_size), points - torch.floor(points), points)
    return torch.where(torch.lt(points, 0), points + torch.ceil(torch.abs(points)), points)
