#!/bin/bash
#PBS -l select=2:ncpus=20:mpiprocs=4
#PBS -l place=scatter
#PBS -N hw6_13_ctest
#PBS -q ctest
#PBS -P ACD107076
#PBS -j oe

module purge
module load pgi/17.10
module load openmpi/pgi/2.1.2/2017
module list

cd $PBS_O_WORKDIR

mpirun -np 1 chp6_13.exe 500 5
mpirun -np 2 chp6_13.exe 500 5
mpirun -np 3 chp6_13.exe 500 5
mpirun -np 4 chp6_13.exe 500 5
mpirun -np 5 chp6_13.exe 500 5
mpirun -np 6 chp6_13.exe 500 5
mpirun -np 7 chp6_13.exe 500 5
mpirun -np 8 chp6_13.exe 500 5
