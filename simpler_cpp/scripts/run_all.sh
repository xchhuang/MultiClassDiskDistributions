#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH -o slurm-%j.out
#SBATCH --gres gpu:1

# run in build folder
./DiskProject ../configs/constrained_overlap.txt
./DiskProject ../configs/constrained.txt
./DiskProject ../configs/forest_no_interaction.txt
./DiskProject ../configs/forest.txt
./DiskProject ../configs/praise_the_sun.txt
./DiskProject ../configs/zerg_rush.txt


