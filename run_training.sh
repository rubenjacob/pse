#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:59:00
#SBATCH --mem=94gb
#SBATCH --partition=gpu_8
#SBATCH --job-name=ruben_pse
#SBATCH --gres=gpu:1

#SBATCH -D /pfs/data5/home/kit/anthropomatik/ht9329/pse
#SBATCH -o /pfs/data5/home/kit/anthropomatik/ht9329/pse/slurm_outputs/slurmlog/out_%A_%a.log
#SBATCH -e /pfs/data5/home/kit/anthropomatik/ht9329/pse/slurm_outputs/slurmlog/err_%A_%a.log

poetry run python3 pse/run.py