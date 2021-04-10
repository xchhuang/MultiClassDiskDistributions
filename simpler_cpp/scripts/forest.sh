#!/bin/bash
#SBATCH -p cpu20
#SBATCH -t 24:00:00
#SBATCH -o slurm-%j.out

# run in build folder
./DiskProject ../configs/forest.txt
cd ..
python plot.py
