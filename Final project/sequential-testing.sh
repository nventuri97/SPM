#!/bin/sh
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -o ./%j-seq.log
#SBATCH -e ./%j-seq.err
#SBATCH -t 02:00:00

echo "Test executed on: $SLURM_JOB_NODELIST"
make SequentialUTWavefront
make run_multiple_tests FILE="SequentialUTWavefront.o" ARGS=$1
echo "done"