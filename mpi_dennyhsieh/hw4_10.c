#include "mpi.h"
#include <stdio.h>

/**
 * Count the number of different six-digit combinations of the numerals 0-9 with 3 constraints:
 * • The first digit may not be a 0.
 * • Two consecutive digits may not be the same.
 * • The sum of the digits may not be 7, 11, or 13.
 * @author DennyHsieh(Tung-Ching Hsieh)
 */
int seperateNum (int n) {
    int i;
    int Value = n;
    int num[6];
    int temp = Value;
    fflush (stdout);

    for (i=0;i<6;i++){
        num[i] = temp%10;
        temp = temp/10;
    }

    int sum = (num[0]+num[1]+num[2]+num[3]+num[4]+num[5]);

    if (!((num[5] == 0) || (num[5] == num[4]) || (num[4] == num[3]) || (num[3] == num[2]) || (num[2] == num[1]) || (num[1] == num[0]) || ((sum == 7) || (sum == 11) || (sum == 13)))) {
        // printf("%d\n", n);
        return 1;
    }else return 0;
}

int main (int argc, char *argv[]) {
    int count;            /* Solutions found by this proc */
    double elapsed_time;  /* Time to find, count solutions */
    int global_count;     /* Total number of solutions */
    int i;
    int id;               /* Process rank */
    int p;                /* Number of processes */
    int seperateNum (int);

    MPI_Init (&argc, &argv);

    /* Start timer */
    MPI_Barrier (MPI_COMM_WORLD);
    elapsed_time = - MPI_Wtime();

    MPI_Comm_rank (MPI_COMM_WORLD, &id);
    MPI_Comm_size (MPI_COMM_WORLD, &p);

    count = 0;
    for (i = id; i < 1000000; i += p)
        count += seperateNum (i);

    MPI_Reduce (&count, &global_count, 1, MPI_INT, MPI_SUM, 0,
        MPI_COMM_WORLD); 

    /* Stop timer */
    elapsed_time += MPI_Wtime();

    if (!id) {
        printf ("The program spends %8.6f seconds\n", elapsed_time);
        fflush (stdout);
    }
    MPI_Finalize();
    if (!id) printf ("There are %d different solutions\n",
        global_count);
    return 0;
}
