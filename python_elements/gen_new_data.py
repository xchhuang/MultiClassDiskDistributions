import numpy as np
import glob
import matplotlib.pyplot as plt
from utils import colors_dict


def main():
    file_path = './examples/zerg_rush.txt'
    target_num_classes = 1
    output_filename = 'first'
    disks = []
    id_map = {}
    with open(file_path) as f:
        lines = f.readlines()
        firstline = lines[0].strip().split(' ')
        nums = [int(x) for x in firstline]
        num_classes = nums[0]
        class_index = nums[1:]
        for i in range(num_classes):
            id_map[class_index[i]] = i

        for line in lines[1:]:
            line = line.strip().split(' ')
            line = [int(x) for x in line]
            c, x, y, r = line
            if r / 10000.0 > 0.01:  # filter small radii, which are not seen properly
                # print(r / 10000.0)
                k = id_map[c]
                k = np.random.choice(target_num_classes)
                disks.append([k, x / 10000.0, y / 10000.0, r / 10000.0])

    disks = np.array(disks)
    # fig, ax = plt.subplots()
    # for i in range(disks.shape[0]):
    #     circle = plt.Circle((disks[i, 1], disks[i, 2]), disks[i, 3], color=colors_dict[disks[i, 0]], fill=False)
    #     ax.add_artist(circle)
    # plt.axis('equal')
    # # plt.xlim([-0.5, 2.5])
    # # plt.ylim([-0.5, 2.5])
    # plt.xlim([-0.2, 1.2])
    # plt.ylim([-0.2, 1.2])
    # plt.title(disks.shape[0])
    # plt.show()

    with open('./examples/'+output_filename+'.txt', 'w') as f:
        f.write(str(target_num_classes))
        for c in range(target_num_classes):
            f.write(' ' + str((c + 1) * 1000))
        f.write('\n')
        for disk in disks:
            c, x, y, r = disk
            f.write(str((int(c) + 1) * 1000) + ' ' + str(int(x * 10000)) + ' ' + str(int(y * 10000)) + ' ' + str(int(r * 10000)) + '\n')

    with open('./configs/'+output_filename+'.txt', 'w') as f:
        f.write('examples/'+output_filename+'.txt'+'\n')
        f.write('0\n')

if __name__ == '__main__':
    main()
