#!/bin/bash
#SBATCH -p cpu20
#SBATCH -t 5:00:00
#SBATCH -o slurm-%j.out

# run in build folder
./DiskProject ../configs/constrained_overlap.txt
./DiskProject ../configs/constrained.txt
./DiskProject ../configs/forest.txt
./DiskProject ../configs/praise_the_sun.txt
./DiskProject ../configs/zerg_rush.txt

cd ..
python plot.py

