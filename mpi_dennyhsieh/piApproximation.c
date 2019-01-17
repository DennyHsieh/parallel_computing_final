#include "mpi.h"
#include <stdio.h>

int main(int argc , char** argv) {
    
    const long partition = 1000000; /* The number of intervals. */
    double interval = 1.0/partition;/* The width of an interval. */
    double sum = 0; /* The accumulated sum of the mini squares. */
    double globalSum = 0;/* The total sum of the mini squares. */
    
    double elapsed_time = 0;/* The time for program to process data. */
    int numprocs; /* The number of processes. */
    int myid; /* The rank of the current process. */
    
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    
    MPI_Barrier (MPI_COMM_WORLD);
    elapsed_time = - MPI_Wtime();

    /* Each process stacks up the square area of its own fraction. */
    for(double iter = interval/2+(myid)*interval ; iter<=1 ;iter += interval*numprocs ) {
        sum += interval*(4/(1+pow(iter,2)));
    }
    
    MPI_Reduce (&sum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    elapsed_time +=  MPI_Wtime();
    
    MPI_Finalize();
    
    if(myid == 0) {
        printf("the approximate sum is %.6f\n",globalSum);
        printf ("The program spend %f seconds\n.",elapsed_time);
    }
}
