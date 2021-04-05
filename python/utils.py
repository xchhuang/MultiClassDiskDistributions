import numpy as np
import torch


def diskDistance(a, b, rmax):
    n_dim = a.size(-1)
    # print(a.shape)
    # print(b.shape)
    radius = torch.cat((a[:, n_dim - 1:n_dim], b[:, n_dim - 1:n_dim]), dim=1)
    # print(radius)
    r1 = torch.max(radius, dim=1)[0] / rmax
    r2 = torch.min(radius, dim=1)[0] / rmax
    # print(r1, r2)
    d = torch.norm(a[:, 0:2] - b[:, 0:2], dim=1) / rmax
    extent = torch.max(d + r1 + r2, 2 * r1)
    overlap = torch.clamp(r1 + r2 - d, 0)
    idx = overlap > 2 * r2
    overlap[idx] = 2 * r2[idx]
    f = extent - overlap + d + r1 - r2
    # print(f)

    idx1 = d <= r1 - r2
    d_norm = torch.zeros_like(d)

    # print (d.size())
    d_norm[idx1] = f[idx1] / (4 * r1[idx1] - 4 * r2[idx1])
    # print (d[idx1].size())

    idx2 = d > r1 - r2
    idx3 = d <= r1 + r2
    idx23 = idx2 * idx3
    # print (d[idx23].size())
    d_norm[idx23] = f[idx23] - 4 * r1[idx23] + 7 * r2[idx23] / (3 * r2[idx23])

    idx4 = d > r1 + r2
    # print (d[idx4].size())
    d_norm[idx4] = f[idx4] - 4 * r1[idx4] - 2 * r2[idx4] + 3

    return d_norm
