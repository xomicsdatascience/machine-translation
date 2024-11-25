#!/bin/bash
# 
#SBATCH -p gpu # partition (queue)
#SBATCH --gpus=1
#SBATCH -c 1 # number of cores
#SBATCH --mem 40G # memory pool for all cores
#SBATCH -t 29-00:00 # time (D-HH:MM)
#SBATCH --job-name=original-transformer
#SBATCH -o /home/cranneyc/machine-translation/scripts/slurmOutputs/slurm.%j.out # STDOUT
#SBATCH -e /home/cranneyc/machine-translation/scripts/slurmOutputs/slurm.%j.err # STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=caleb.cranney@cshs.org


echo "machine-translation start"
eval "$(conda shell.bash hook)"
conda activate machine-translation
cd /home/cranneyc/machine-translation/scripts 
python 1_train_model.py
