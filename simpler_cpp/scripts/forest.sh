#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH -o slurm-%j.out
#SBATCH --gres gpu:1

# run in build folder
./DiskProject ../configs/forest.txt
cd ..
python plot.py
