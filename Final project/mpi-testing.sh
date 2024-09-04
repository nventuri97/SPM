#!/bin/sh
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=8                  # Number of MPI processes (distributed per node)
#SBATCH --cpus-per-task=1           # Number of CPUs per task (=thread)
#SBATCH -o ./%j-mpi.log
#SBATCH -e ./%j-mpi.err
#SBATCH -t 02:00:00

echo "Test executed on: $SLURM_JOB_NODELIST with $SLURM_NTASKS"
make $1
make run_multiple_tests FILE="$1.o" ARGS="$2"
echo "done"