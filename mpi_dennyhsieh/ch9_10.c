#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define NEW_ASSIGN 0
#define FIND_PRIME 1
#define SEND_NUMBER 2
#define TERMINATE 3

/**
 * A parallel program to find the first eight perfect numbers.
  * @author DennyHsieh(Tung-Ching Hsieh), LiangChiaLun
 */
int isPrime (long long unsigned input) {
    long long unsigned root = (long long unsigned)sqrt(input);
    for (int i = 2 ; i <= root ; i++) {
        if (input % i == 0)
            return 0;
        else
            continue;
    }
    return 1;
}


int main (int argc, char *argv[]) {

    double elapsed_time = 0;
    int num_proc, my_id;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = - MPI_Wtime();
    
    if (my_id == 0) {

        int assigned_to = 2;
        int findNumber = 0;
        int* record = (int*)malloc(sizeof(int)*8);
        int* num = (int*)malloc(sizeof(int));
        int source;
        while (1) {


            MPI_Recv(num,1,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            source = status.MPI_SOURCE;

            if (status.MPI_TAG == FIND_PRIME) {
                record[findNumber] = *num;
                findNumber++;

                if (findNumber == 8) {
                    MPI_Send (num, 1, MPI_INT,source,TERMINATE, MPI_COMM_WORLD);
                    for(int i = 7 ; i > 0  ; i--) {
                        for(int j = 0 ; j < i ; j++) {
                            if (record[j] > record[j + 1]) {
                                int tmp;
                                tmp = record[j];
                                record[j] = record[j+1];
                                record[j+1] = tmp;
                            }
                        }
                    }
                    for(int i = 1 ; i < num_proc ; i++) {
                        if(i != source) {
                            MPI_Recv(num, 1, MPI_INT,i,MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                            if(status.MPI_TAG == FIND_PRIME) {
                                int position = 0;
                                for(int j = 0 ; j < 8 ; j++) {
                                    if (*num > record[j]) {
                                        position++;
                                    }
                                    else
                                        break;
                                }
                                for (int i = position ; i < 7 ; i++) {
                                    record[i+1] = record[i];
                                }
                                record[position] = *num;
                            }
                            MPI_Send (num, 1,MPI_INT,i,TERMINATE, MPI_COMM_WORLD);
                        }
                    }
                    int size = 0;
                    for(int i = 0; i < 10 ; i++) {
                        if (record[i] != 0)
                            size++;
                    }
                    for (int i = 0 ; i < 8 ; i++) {
                        printf("for prime %d ,%llu is a perfect number \n",record[i],(long long unsigned)pow(2,record[i]-1)*(long long unsigned)(pow(2,record[i])-1));
                    }
                    break;
                }
                else {
                    int* assigned_num = (int*)malloc(sizeof(int));
                    *assigned_num =  assigned_to++;
                    MPI_Send (assigned_num, 1, MPI_INT,source, SEND_NUMBER, MPI_COMM_WORLD);
                }
            }
            else if (status.MPI_TAG == NEW_ASSIGN) {
                int* assigned_num = (int*)malloc(sizeof(int));
                *assigned_num = assigned_to++;
                MPI_Send (assigned_num, 1, MPI_INT,source, SEND_NUMBER, MPI_COMM_WORLD);
            }
        }
    }
    else if (my_id != 0) {
        int* numRec =  (int*)malloc(sizeof(int));
        long long unsigned checkNum = 0;
         MPI_Send (numRec, 1, MPI_INT,0, NEW_ASSIGN, MPI_COMM_WORLD);
        while (1) {
            MPI_Recv(numRec,1,MPI_INT,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            if (status.MPI_TAG == TERMINATE) {
                break;
            }
            checkNum = (long long unsigned)(pow(2,*numRec) - 1);
            if (isPrime(checkNum)) {
                MPI_Send (numRec, 1, MPI_INT,0, FIND_PRIME, MPI_COMM_WORLD);
            }
            else {
                MPI_Send (numRec, 1, MPI_INT,0, NEW_ASSIGN, MPI_COMM_WORLD);
            }
        }
    }
    elapsed_time +=  MPI_Wtime();
    if(my_id == 0) {
        printf ("The program spends %f seconds\n.",elapsed_time);
    }
    MPI_Finalize();
}


