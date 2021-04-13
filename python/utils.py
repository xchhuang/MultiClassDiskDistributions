import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from graphlib import TopologicalSorter


class Contribution:
    def __init__(self, device, nbbins):
        self.weights = torch.zeros(nbbins).float().to(device)
        self.contribution = torch.zeros(nbbins).float().to(device)
        self.pcf = torch.zeros(nbbins).float().to(device)


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
