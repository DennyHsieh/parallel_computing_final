#include "MyMPI.h" 
#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

/**
 * Show a live cell parallel program that reads from a fiie an m*n matrix
 * @author LiangChiaLun, DennyHsieh(Tung-Ching Hsieh)
 */
int main(int argc,char *argv[])
{

    int j = atoi(argv[1]);
    int k = atoi(argv[2]);
    int myid;/* The rank of the process. */
    int numprocs;/* The number of the processed */
    
    int** a;         /* Doubly-subscripted array */
    int*  storage;   /* Local portion of array elements */
    
    int     m;         /* Rows in matrix */
    int     n;         /* Columns in matrix */
    
    int* size_low;
    int* size_high;
    int* size_proc;
    
    int top_process = -1;
    int down_process = -1;
    int size = -1;
    
    int* top_array;
    int* down_array;
    
    double elapsed_time = 0;
    
    MPI_Status status;
    
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Barrier (MPI_COMM_WORLD);
    elapsed_time = - MPI_Wtime();

    size_low =  malloc(numprocs*sizeof( int));
    size_high = malloc(numprocs*sizeof( int));
    size_proc = malloc(numprocs*sizeof( int));  
    
    // read_row_striped_matrix ("/Users/kuipasta1121/Desktop/parallel1/file.bin", (void *) &a,
    read_row_striped_matrix ("/work1/bonjour22889/hw/hw6/hw6_13/file.bin", (void *) &a,
                            (void *) &storage, MPI_INT, &m, &n, MPI_COMM_WORLD) ;
    
    top_array = malloc(n*sizeof( int));
    down_array = malloc(n*sizeof( int));
    
    if(myid==0){
        for(int i=0 ; i<numprocs ;i++){
           *(size_proc+i) = BLOCK_SIZE(i,numprocs,m);
           *(size_low+i) =  BLOCK_LOW(i,numprocs,m);
           *(size_high+i) =  BLOCK_HIGH(i,numprocs,m);
        }
    }
                        
    MPI_Bcast(size_low, numprocs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(size_proc, numprocs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(size_high, numprocs, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Barrier (MPI_COMM_WORLD);
    
    if(size_proc[myid] != 0) {
        if(size_low[myid]!=0) {
            for(int i=0 ; i<numprocs ; i++){
                if(size_high[i] == size_low[myid]-1 && size_proc[i]!=0)
                    top_process = i;
            }
        }
        if(size_high[myid] != m-1) {
            for(int i=0 ; i<numprocs ; i++){
                if(size_low[i] == size_high[myid]+1 && size_proc[i]!=0)
                    down_process = i;
            }
        }
    }
    size = size_proc[myid];
    
    free(size_proc);
    free(size_low);
    free(size_high);
    
    if(size!=0) {
        print_row_striped_matrix_dot ((void**)a, MPI_INT, m, n,MPI_COMM_WORLD);
        row_transfer(n, size, top_process ,down_process ,top_array ,down_array, a, status);
        for(int i=0 ; i<j ; i++) {
            update(n, size, top_process, down_process, top_array, down_array, a);
            row_transfer(n,size, top_process , down_process , top_array ,  down_array,a,status);
            if((i+1)%k == 0) {
                if(myid==0)
                    printf("this is iteration %d:\n\n",i+1);
                print_row_striped_matrix_dot ((void**)a, MPI_INT, m, n,MPI_COMM_WORLD);
            }
        }
        
    }
    elapsed_time +=  MPI_Wtime();
    if(myid == 0) {
        printf ("The program spends %f seconds\n.",elapsed_time);
    }
    MPI_Finalize();
}
