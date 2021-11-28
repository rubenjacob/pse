#!/bin/bash

#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH -t 23:59:00
#SBATCH --mem=20gb
#SBATCH -p gpu_8
#SBATCH -J  ruben_pse_baseline
#SBATCH --gres=gpu:1

#SBATCH -D /pfs/data5/home/kit/anthropomatik/ht9329/pse
#SBATCH -o /pfs/data5/home/kit/anthropomatik/ht9329/pse/slurm_outputs/slurmlog/out_%A_%a.log
#SBATCH -e /pfs/data5/home/kit/anthropomatik/ht9329/pse/slurm_outputs/slurmlog/err_%A_%a.log

poetry run python3 pse/run.py