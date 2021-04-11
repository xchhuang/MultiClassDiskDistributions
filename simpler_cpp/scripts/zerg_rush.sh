#!/bin/bash
#SBATCH -p cpu20
#SBATCH -t 24:00:00
#SBATCH -o slurm-%j.out

# run in build folder
./DiskProject ../configs/zerg_rush.txt
cd ..
python plot.py
