#!/bin/bash

#SBATCH -p preemptable,gpu # partition (queue)
#SBATCH --gpus=1
#SBATCH -c 1 # number of cores
#SBATCH --mem 99G # memory pool for all cores
#SBATCH -t 2-23:00 # time (D-HH:MM)
#SBATCH --job-name=mt_small
#SBATCH -o /home/cranneyc/machine-translation/scripts/slurmOutputs/slurm.%j.out # STDOUT
#SBATCH -e /home/cranneyc/machine-translation/scripts/slurmOutputs/slurm.%j.err # STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=caleb.cranney@cshs.org

echo "machine-translation start"
eval "$(conda shell.bash hook)"
conda activate machine-translation
cd /home/cranneyc/machine-translation/scripts

# Check if 3 command-line arguments are provided
if [ $# -ne 3 ]; then
  echo "Usage: sbatch script.sh <arg1> <arg2> <arg3>"
  exit 1
fi

# Run the Python script with the provided arguments
python 1_train_model__small.py "$1" "$2" "$3"
