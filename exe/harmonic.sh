#!/bin/bash
#PBS -l select=2:ncpus=20:mpiprocs=4
#PBS -l place=scatter
#PBS -N harmonic_1_ctest
#PBS -q ctest
#PBS -P ACD107076
#PBS -j oe

module purge
module load pgi/17.10
module load openmpi/pgi/2.1.2/2017
module list

cd $PBS_O_WORKDIR

mpirun -np 1 harmonic.exe
