#!/bin/bash
#SBATCH -p cpu20
#SBATCH -t 1:00:00
#SBATCH -o slurm_outputs/slurm-%j.out

mkdir -p slurm_outputs

python main.py --config_filename=configs/zerg_rush.txt
