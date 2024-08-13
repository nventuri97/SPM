#!/bin/sh
#SBATCH -p normal
#SBATCH -N 4
#SBATCH --ntasks=8                  # Number of MPI processes (distributed per node)
#SBATCH --cpus-per-task=1           # Number of CPUs per task (=process)
#SBATCH -o ./%j-mpi.log
#SBATCH -e ./%j-mpi.err
#SBATCH -t 02:00:00

echo "Test executed on: $SLURM_JOB_NODELIST"
make broad
mpirun ./BroadMPIUTWavefront.o $1 $2
echo "done"