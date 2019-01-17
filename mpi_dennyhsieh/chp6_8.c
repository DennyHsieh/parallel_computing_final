#include "mpi.h"
#include <stdio.h>
#include <math.h>

/**
 * The time needed to send an n-byte message is lambda + n/beta. 
 * Do the "ping pong" test to determine X (latency) and f (bandwidth) on parallel computer
 * @author LiangChiaLun, DennyHsieh(Tung-Ching Hsieh)
 */
int main(int argc,char *argv[])
{
    int myid;/* The rank of the process. */
    int numprocs;/* The number of the processed */
    
    int message_length = 20;/* The message length in byte */
    int passing_time = 10000000;/* Message passing time */
    
    char msg[message_length];/* Declaration of the sending message with specified message length. */
    double elapsed_time[10];/* The time for program to process data. */
    
    MPI_Status status;
    
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Barrier (MPI_COMM_WORLD);
    
    for (int i=0 ; i<10 ; i++) {
        
        if(myid == 0) {
            elapsed_time[i] = 0;
            elapsed_time[i] -= MPI_Wtime();
            
            for(int j=0 ; j++ ; j<passing_time) {
                /*For process 0, it sends the message to process 1 first and then receive the message sent back from process 1*/
                MPI_Send(&msg, message_length, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&msg, message_length, MPI_BYTE, 1, 0, MPI_COMM_WORLD, &status);
            }
            elapsed_time[i] += MPI_Wtime();
        }
    
        if(myid == 1) {
        
             for(int j=0 ; j++ ; j<passing_time) {
                 /*For process 1, it receives the message from process 0 first and then immediately send it back to process 0*/
                 MPI_Recv(&msg, message_length, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
                 MPI_Send(&msg, message_length, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            }
        }
    }
    
    
    if(myid == 0) {
        
        double avg_time = 0;
        for (int i=0 ; i<10 ; i++) {
            avg_time += elapsed_time[i];
        }
        avg_time /= 10;
        printf("It takes in average %.10f seconds for trying sending the message %d time with length %d byte.\n",avg_time,passing_time,message_length);
    }
    MPI_Finalize();
}
