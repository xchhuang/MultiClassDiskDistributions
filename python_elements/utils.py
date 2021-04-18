import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from graphlib import TopologicalSorter
import sys

# newly added
sys.path.append('../')
from common.fps import farthest_point_sample
from sklearn.cluster import KMeans
from common import smallestenclosingcircle


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


def plot_elements(topological_order, outputs, filename):
    fig, ax = plt.subplots()
    for k in topological_order:
        out = outputs[k]
        out = np.array(out)
        for i in range(out.shape[0]):
            for j in range(1, out.shape[1]):
                # plt.scatter(out[:, 0], out[:, 1], s=5)
                circle = plt.Circle((out[i, j, 0], out[i, j, 1]), out[i, j, 2], color=colors_dict[k], fill=False)
                ax.add_artist(circle)
        plt.axis('equal')
        plt.xlim([-0.2, 1.2])
        plt.ylim([-0.2, 1.2])
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


def getSamplesFromImage_helper(coord, samples_per_element):
    coord_sample1 = farthest_point_sample(coord, npoint=1024)
    if samples_per_element > 1:
        coord_sample2 = farthest_point_sample(coord_sample1, npoint=samples_per_element)
    else:
        coord_sample2 = np.expand_dims(np.mean(coord_sample1, 0), 0)
    # sample_center = np.mean(coord_sample2, 0)
    # plt.figure(1)
    # plt.scatter(coord_sample1[:, 0], coord_sample1[:, 1], c='r', s=2)
    # plt.scatter(coord_sample2[:, 0], coord_sample2[:, 1], c='b', s=20)
    # plt.show()

    kmeans = KMeans(n_clusters=samples_per_element, init=coord_sample2, n_init=1, max_iter=1).fit(coord_sample2)
    pred_sample1 = kmeans.predict(coord_sample1)

    sample_spheres = []

    if samples_per_element == 1:
        actual = smallestenclosingcircle.make_circle(coord_sample1)
        x, y, r = actual
        sample_spheres.append([x, y, r])
    else:
        for s in range(samples_per_element):
            idx = pred_sample1 == s
            cur_s = coord_sample1[idx]
            actual = smallestenclosingcircle.make_circle(cur_s)
            x, y, r = actual
            sample_spheres.append([x, y, r])

    sample_spheres = np.array(sample_spheres)
    # sample_spheres_copy = sample_spheres.copy()
    # # print (sample_spheres.shape,np.mean(sample_spheres_copy,0).shape)
    # sample_spheres[:, 0:2] = sample_spheres_copy[:, 0:2] - np.expand_dims(
    #     np.mean(sample_spheres_copy[:, 0:2], 0), 0)
    # print(sample_spheres.shape, sample_spheres)
    return sample_spheres


def getSamplesFromImage(im, samples_per_element):
    """
    :param img: ndarray: [W, H, 3or4]
    :param samples_per_element: >1
    :return:
    """

    im = np.mean(im, -1)
    coord = np.argwhere(im > 0) / im.shape[0]
    coord = np.flip(coord, axis=1)
    coord[:, 1] = 1 - coord[:, 1]
    # print(im.shape)
    # plt.figure(1)
    # plt.scatter(coord[:, 0], coord[:, 1])
    # plt.show()

    sample_spheres1 = getSamplesFromImage_helper(coord, 1)
    sample_spheres2 = getSamplesFromImage_helper(coord, samples_per_element)
    # print(sample_spheres1.shape, sample_spheres2.shape)
    c = sample_spheres1[0, 0:2]

    # print('before:', sample_spheres2)
    sample_spheres_center = sample_spheres1.copy()
    sample_spheres_elem = sample_spheres2.copy()
    sample_spheres_center[:, 0:2] = sample_spheres1[:, 0:2] - c
    sample_spheres_elem[:, 0:2] = sample_spheres2[:, 0:2] - c
    # print('after:', sample_spheres2)
    # plt.figure(1)
    # for ss in range(len(sample_spheres_center)):
    #     x, y, r = sample_spheres_center[ss]
    #     circle = plt.Circle((x, y), r, color='g', fill=False)
    #     plt.gcf().gca().add_artist(circle)
    # for ss in range(len(sample_spheres_elem)):
    #     x, y, r = sample_spheres_elem[ss]
    #     circle = plt.Circle((x, y), r, color='g', fill=False)
    #     plt.gcf().gca().add_artist(circle)
    # plt.scatter(sample_spheres_center[0, 0], sample_spheres_center[0, 1], c='r', s=5)
    # plt.axis('equal')
    # plt.xlim([-1, 1])
    # plt.ylim([-1, 1])
    # plt.show()
    # print(sample_spheres_center.shape, sample_spheres_elem.shape)
    sample_spheres = np.concatenate([sample_spheres_center, sample_spheres_elem], 0)
    return sample_spheres
