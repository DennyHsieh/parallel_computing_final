#!/bin/bash
#PBS -l select=2:ncpus=20:mpiprocs=4
#PBS -l place=scatter
#PBS -N hw4_12_ctest
#PBS -q ctest
#PBS -P ACD107076
#PBS -j oe

module purge
module load pgi/17.10
module load openmpi/pgi/2.1.2/2017
module list

cd $PBS_O_WORKDIR

mpirun -np 1 ch4_12.exe
mpirun -np 2 ch4_12.exe
mpirun -np 3 ch4_12.exe
mpirun -np 4 ch4_12.exe
mpirun -np 5 ch4_12.exe
mpirun -np 6 ch4_12.exe
mpirun -np 7 ch4_12.exe
mpirun -np 8 ch4_12.exe
