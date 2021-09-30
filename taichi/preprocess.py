import numpy as np
import matplotlib.pyplot as plt

scene_name = 'hexa'
example_filename = 'examples/' + scene_name + '.txt'
edge_space = 0.09


def normalization_np(points, d_space=2, edge_space=0.02):
    if edge_space == 0:
        return points
    if not isinstance(points, np.ndarray):
        points = points.numpy()
    else:
        points = points.copy()
    # print (points.shape)
    # print (points.shape)
    min = points.min(0)
    max = points.max(0)
    # print(min, max)
    # print (min.shape)
    # min = points.min(0)[0]
    # max = points.max(0)[0]
    for id in range(d_space):
        r = max[id] - min[id]
        points[:, id] = (((points[:, id] - min[id]) / r) * (1 - 2 * edge_space) + edge_space)
    return points


pts = []
cls = []
rad = []

num_classes = 0
with open(example_filename) as e:
    ef = e.readlines()
    firstline = []
    for line in ef[0:1]:
        firstline = line.strip().split(' ')
        firstline = [int(x) for x in firstline]
        # print(firstline)
        num_classes = firstline[0]
    for line in ef[1:]:
        line = line.strip().split(' ')
        c = int(line[0])
        x, y, r = line[1], line[2], line[3]
        x, y, r = int(x) / 10000.0, int(y) / 10000.0, int(r) / 10000.0
        pts.append([x, y])
        cls.append(c)
        rad.append(r)

ori_pts = np.array(pts)
cls = np.array(cls)
rad = np.array(rad)
# print(ori_pts.shape, cls.shape, rad.shape)

pts = normalization_np(ori_pts, edge_space=edge_space)

# plt.figure(1)
# plt.subplot(121)
# plt.scatter(ori_pts[:, 0], ori_pts[:, 1])
# plt.gca().set_aspect('equal')
# plt.subplot(122)
# plt.scatter(pts[:, 0], pts[:, 1])
# plt.gca().set_aspect('equal')
# plt.show()


with open('examples/{:}_pp.txt'.format(scene_name), 'w') as f:
    f.write(str(num_classes))
    for i in range(num_classes):
        f.write(' ' + str((i + 1) * 1000))
    f.write('\n')
    for i in range(pts.shape[0]):
        # f.write(str(out[i, 0]) + ' ' + str(out[i, 1]) + '\n')
        c = cls[i]
        r = rad[i]
        f.write(str(c) + ' ' + str(int(pts[i, 0] * 10000)) + ' ' + str(int(pts[i, 1] * 10000)) + ' ' + str(int(r * 10000)) + '\n')
