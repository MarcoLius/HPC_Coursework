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

# Make an array of the number of threads
declare thread_counts=(1 2 4 8 16)

# Declare a file to store the results
output_file="time_results.txt"

# Create the output file
> $output_file

# Use for loop to test the time with different number of threads
for threads in "${thread_counts[@]}"; do
    export OMP_NUM_THREADS=$threads
    echo "Running with $threads threads" >> $output_file
    /usr/bin/time -o $output_file -a ./part1
done

# run part1 with 4 threads
#export OMP_NUM_THREADS=4

# Test the time of running
#time ./part1
