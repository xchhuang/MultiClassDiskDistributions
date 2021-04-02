import numpy as np
# import torch
import matplotlib.pyplot as plt
import math
import glob

def read_txt(path='build/init_pts_final.txt', start=1, normalize=True):
    pts = []
    cla = []
    with open(path) as f:
        lines = f.readlines()
        lines = lines[start:]
        for line in lines:
            line = line.strip().split()
#            if normalize:
            c, x, y, r = line
            x = float(x) / 10000
            y = float(y) / 10000
            c = int(c)
#            else:
#                x, y, c, r = line
#                x = float(x)
#                y = float(y)
#                c = int(c)
            pts.append([x, y])
            cla.append(c)

    pts = np.array(pts)
    cla = np.array(cla)
    return cla, pts


def read_all_txt(path='outputs'):
    files = glob.glob(path+'/*.txt')
    for file in files:
        scene_name = file.split('/')[-1].split('.')[0].split('_')
        scene_name = '_'.join(scene_name[:-1])
        print(scene_name)
        init_cla, init_pts = read_txt(path='outputs/'+scene_name+'_init.txt', start=1, normalize=False)
        tar_cla, tar_pts = read_txt(path='examples/'+scene_name+'.txt', start=1, normalize=True)
        plt.figure(1)
        plt.subplot(121)
        plt.title('Synthesized')
        plt.scatter(init_pts[:, 0], init_pts[:, 1], c=init_cla, s=5)
        plt.axis('equal')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.subplot(122)
        plt.title('Input Exemplar')
        plt.scatter(tar_pts[:, 0], tar_pts[:, 1], c=tar_cla, s=5)
        plt.axis('equal')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig('outputs/'+scene_name)
        plt.clf()


def plot_txt():
    scene_name = 'zerg_rush'
    init_cla, init_pts = read_txt(path='outputs/'+scene_name+'_init.txt', start=1, normalize=False)
    tar_cla, tar_pts = read_txt(path='examples/toy_3.txt', start=1, normalize=True)

    plt.figure(1)
    plt.subplot(121)
    plt.title('Synthesized')
    plt.scatter(init_pts[:, 0], init_pts[:, 1], c=init_cla, s=5)
    plt.axis('equal')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.subplot(122)
    plt.title('Input Exemplar')
    plt.scatter(tar_pts[:, 0], tar_pts[:, 1], c=tar_cla, s=5)
    plt.axis('equal')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig('outputs/result')
    plt.clf()


def main():
    read_all_txt()


if __name__ == '__main__':
    main()





