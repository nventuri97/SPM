#!/bin/sh
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -o ./%j-ff.log
#SBATCH -e ./%j-ff.err
#SBATCH -t 02:00:00

echo "Test executed on: $SLURM_JOB_NODELIST"
make FFUTWavefront
make run_multiple_tests FILE="FFUTWavefront.o" ARGS="$1 $2"
echo "done"