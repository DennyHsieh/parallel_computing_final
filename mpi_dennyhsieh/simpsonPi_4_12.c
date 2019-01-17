#include "mpi.h"
#include <stdio.h>
#include <math.h>
/* Function of 4/1+x^2. */

#define partition 50 /* The number of intervals. */

/**
 * Compute the value of 𝜋 using Simpson's Rule
 * @author LiangChiaLun, DennyHsieh(Tung-Ching Hsieh)
 */
double f (int i) {
    double x;
    x = i*1.0 / partition;
    return 4.0 / (1.0 + x*x ) ;
}


int main(int argc , char** argv) {
    
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
    
    if(myid == 0) {
        sum += (f(0) - f(partition));
    }
    /* Each process stacks up the square area of its own fraction. */
    for(int i= 1+myid ; i <= partition/2 ; i += numprocs) {
        sum += 4.0*f(2*i-1) + 2*f(2*i);
    }
    sum /= (3.0* partition);
    // printf("this is process %d and the sum is %.6f\n",myid,sum);
    MPI_Reduce (&sum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    elapsed_time +=  MPI_Wtime();
    
    MPI_Finalize();
    
    if(myid == 0) {
        printf("the approximate sum is %.6f\n",globalSum);
        printf ("The program spend %f seconds\n.",elapsed_time);
    }
}

