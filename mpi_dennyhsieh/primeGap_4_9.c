#include "mpi.h" 
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

/**
 Check if an integer is a prime number.
 Returns 1 if the integer is a prime number,
 or otherwise, 0.
 * @author DennyHsieh(Tung-Ching Hsieh), LiangChiaLun
 */
*/
char isPrime(int num) {
    
    /* By definition, 1 is not a prime number. */
    if(num == 1) {
        return 0;
    }
    
    int root = sqrt(num);
    
    /* Find factors for the integer from 2 to the biggest integer under its squareroot. */
    for(int i = 2; i <= root ; i++) {
        
        if(num%i == 0) {
            
            /* Return 0 if any factors are found. */
            return 0;
        }
    }
    
    /* Returns 1 if the integer is a prime number. */
    return 1;
}



int main(int argc, char **argv)
{

    const long limit = 1000000; /* The upper bound for searching the prime gap. */
    int lastPrime = 0; /* The last prime just be found. */
    int gap = 0; /* The biggest prime gap so far. */
    int globalGap = 0; /* The grand biggest prime gap. */
    
    double elapsed_time = 0;/* The time for program to process data. */
    int numprocs; /* The number of processes. */
    int myid; /* The rank of the current process. */
    

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    
    MPI_Barrier (MPI_COMM_WORLD);
    elapsed_time = - MPI_Wtime();

    
    /* Start to search the gap from 2 if the rank is 0.
     Otherwise, just start from the corresponding id/num ratio of the upper bound. */
    int iter = (myid==0)?2:limit*(myid*1.0/numprocs);
    
    for(iter ; iter <= limit*((myid+1)*1.0/numprocs); iter++) {
        
        if(isPrime(iter)) {
            
            if(lastPrime != 0) {
                /* If the new gap is bigger than the older one, update the gap. */
                gap = (iter - lastPrime > gap)?
                            (iter - lastPrime):gap;
            }
            lastPrime = iter;
        }
    }
    
    /* Reduce to the grand biggest prime gap. */
    MPI_Reduce (&gap, &globalGap, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    elapsed_time +=  MPI_Wtime();
    
    MPI_Finalize();
    
    if(myid==0) {
        printf ("The max gap is %d \n",globalGap);
        printf ("The program spend %f seconds\n.",elapsed_time);
    }
    
}

