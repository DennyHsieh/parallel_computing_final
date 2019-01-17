#include "mpi.h"
#include <stdio.h>

/**
 * Let S_n = sigma(1/i), from i = 1 to n
 * Computes sums to arbitrary precision after the decimal point
 * S_1000000 to 100 digits
 * @author DennyHsieh(Tung-Ching Hsieh)
 */
int main (int argc, char *argv[]) {
    int i;
    int id;               /* Process rank */
    int p;                /* Number of processes */
    int n, d;             /* sum from 1/1 to 1/n with d decimal point*/
    double sum;           /* Solutions found by this proc */
    double elapsed_time;  /* Time to find, count solutions */
    double global_sum;    /* Sum of solutions */

    MPI_Init(&argc, &argv);

    /* Start timer */
    MPI_Barrier (MPI_COMM_WORLD);
    elapsed_time = - MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    n = 1000000;
    d = 100;

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    sum = 0;
    for (i = id+1; i <= n; i += p) {
        sum += 1.0/i;
    }
    MPI_Reduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Stop timer */
    elapsed_time += MPI_Wtime();

    if (!id) {
        printf("The sum is %.*f\n", d, global_sum);
        fflush (stdout);
        printf ("The program spends %8.6f seconds\n", elapsed_time);
        fflush (stdout);
    }
    MPI_Finalize();
    return 0;
}