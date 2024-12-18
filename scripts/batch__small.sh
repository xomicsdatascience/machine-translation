#!/bin/bash
# 
#SBATCH -p preemptable,gpu # partition (queue)
#SBATCH --gpus=a100:1
#SBATCH -c 1 # number of cores
#SBATCH --mem 99G # memory pool for all cores
#SBATCH -t 29-23:00 # time (D-HH:MM)
#SBATCH --job-name=mt_small
#SBATCH -o /home/cranneyc/machine-translation/scripts/slurmOutputs/slurm.%j.out # STDOUT
#SBATCH -e /home/cranneyc/machine-translation/scripts/slurmOutputs/slurm.%j.err # STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=caleb.cranney@cshs.org


echo "machine-translation start"
eval "$(conda shell.bash hook)"
conda activate machine-translation
cd /home/cranneyc/machine-translation/scripts 
python 1_train_model__small.py
