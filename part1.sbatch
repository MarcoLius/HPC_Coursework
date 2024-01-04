#!/bin/bash

# Request resources:
#SBATCH --job-name="part1"
#SBATCH -o part1.%J.out
#SBATCH -e part1.%J.err
#SBATCH -N 1		# number of compute nodes. 
#SBATCH -c 4		# number of CPU cores, one per thread, up to 128
#SBATCH --mem=1G	# memory required, up to 250G on standard nodes
#SBATCH --time=0:15:0	# time limit for job (format:  days-hours:minutes:seconds)

# Run in the 'shared' queue (job may share node with other jobs)
#SBATCH -p shared

# Modules necessary for job:
module purge
module load gcc

# compile part1.c into part1
gcc -fopenmp -lm part1.c -o part1

# run part1 with 4 threads
OMP_NUM_THREADS=4 ./part1
