#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=20gb
#SBATCH --partition=single
#SBATCH --job-name=ruben_pse_data_collection

#SBATCH -D /pfs/data5/home/kit/anthropomatik/ht9329/pse
#SBATCH -o /pfs/data5/home/kit/anthropomatik/ht9329/pse/slurm_outputs/slurmlog/out_%A_%a.log
#SBATCH -e /pfs/data5/home/kit/anthropomatik/ht9329/pse/slurm_outputs/slurmlog/err_%A_%a.log

poetry run python3 pse/data/data_collection.py