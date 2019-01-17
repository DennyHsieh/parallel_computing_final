//#include "MyMPI.h" 
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
/************************* MACROS **************************/

#define DATA_MSG           0
#define PROMPT_MSG         1
#define RESPONSE_MSG       2

#define OPEN_FILE_ERROR    -1
#define MALLOC_ERROR       -2
#define TYPE_ERROR         -3

#define MIN(a,b)           ((a)<(b)?(a):(b))
#define BLOCK_LOW(id,p,n)  ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id,p,n) \
                     (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)
#define BLOCK_OWNER(j,p,n) (((p)*((j)+1)-1)/(n))
#define PTR_SIZE           (sizeof(void*))
#define CEILING(i,j)       (((i)+(j)-1)/(j))

void test() {
   printf("test valid");
}
/***************** MISCELLANEOUS FUNCTIONS *****************/

/*
 *   Given MPI_Datatype 't', function 'get_size' returns the
 *   size of a single datum of that data type.
 */

int get_size (MPI_Datatype t) {
   if (t == MPI_BYTE) return sizeof(char);
   if (t == MPI_CHAR) return sizeof(char);
   if (t == MPI_DOUBLE) return sizeof(double);
   if (t == MPI_FLOAT) return sizeof(float);
   if (t == MPI_INT) return sizeof(int);
   printf ("Error: Unrecognized argument to 'get_size'\n");
   fflush (stdout);
   MPI_Abort (MPI_COMM_WORLD, TYPE_ERROR);
   return 0;
}


/*
 *   Function 'my_malloc' is called when a process wants
 *   to allocate some space from the heap. If the memory
 *   allocation fails, the process prints an error message
 *   and then aborts execution of the program.
 */

void *my_malloc (
   int id,     /* IN - Process rank */
   int bytes)  /* IN - Bytes to allocate */
{
   void *buffer;
   if ((buffer = malloc ((size_t) bytes)) == NULL) {
      printf ("Error: Malloc failed for process %d\n", id);
      fflush (stdout);
      MPI_Abort (MPI_COMM_WORLD, MALLOC_ERROR);
   }
   return buffer;
}


/*
 *   Function 'terminate' is called when the program should
 *   not continue execution, due to an error condition that
 *   all of the processes are aware of. Process 0 prints the
 *   error message passed as an argument to the function.
 *
 *   All processes must invoke this function together!
 */

void terminate (
   int   id,            /* IN - Process rank */
   char *error_message) /* IN - Message to print */
{
   if (!id) {
      printf ("Error: %s\n", error_message);
      fflush (stdout);
   }
   MPI_Finalize();
   exit (-1);
}


/************ DATA DISTRIBUTION FUNCTIONS ******************/

/*
 *   This function creates the count and displacement arrays
 *   needed by scatter and gather functions, when the number
 *   of elements send/received to/from other processes
 *   varies.
 */

void create_mixed_xfer_arrays (
   int id,       /* IN - Process rank */
   int p,        /* IN - Number of processes */
   int n,        /* IN - Total number of elements */
   int **count,  /* OUT - Array of counts */
   int **disp)   /* OUT - Array of displacements */
{

   int i;

   *count = my_malloc (id, p * sizeof(int));
   *disp = my_malloc (id, p * sizeof(int));
   (*count)[0] = BLOCK_SIZE(0,p,n);
   (*disp)[0] = 0;
   for (i = 1; i < p; i++) {
      (*disp)[i] = (*disp)[i-1] + (*count)[i-1];
      (*count)[i] = BLOCK_SIZE(i,p,n);
   }
}


/*
 *   This function creates the count and displacement arrays
 *   needed in an all-to-all exchange, when a process gets
 *   the same number of elements from every other process.
 */

void create_uniform_xfer_arrays (
   int id,        /* IN - Process rank */
   int p,         /* IN - Number of processes */
   int n,         /* IN - Number of elements */
   int **count,   /* OUT - Array of counts */
   int **disp)    /* OUT - Array of displacements */
{

   int i;

   *count = my_malloc (id, p * sizeof(int));
   *disp = my_malloc (id, p * sizeof(int));
   (*count)[0] = BLOCK_SIZE(id,p,n);
   (*disp)[0] = 0;
   for (i = 1; i < p; i++) {
      (*disp)[i] = (*disp)[i-1] + (*count)[i-1];
      (*count)[i] = BLOCK_SIZE(id,p,n);
   }
}

/*
 *   This function is used to transform a vector from a
 *   block distribution to a replicated distribution within a
 *   communicator.
 */

void replicate_block_vector (
   void        *ablock,  /* IN - Block-distributed vector */
   int          n,       /* IN - Elements in vector */
   void        *arep,    /* OUT - Replicated vector */
   MPI_Datatype dtype,   /* IN - Element type */
   MPI_Comm     comm)    /* IN - Communicator */
{
   int *cnt;  /* Elements contributed by each process */
   int *disp; /* Displacement in concatenated array */
   int id;    /* Process id */
   int p;     /* Processes in communicator */

   MPI_Comm_size (comm, &p);
   MPI_Comm_rank (comm, &id);
   create_mixed_xfer_arrays (id, p, n, &cnt, &disp);
   MPI_Allgatherv (ablock, cnt[id], dtype, arep, cnt,
                   disp, dtype, comm);
   free (cnt);
   free (disp);
}

/********************* INPUT FUNCTIONS *********************/

/*
 *   Function 'read_checkerboard_matrix' reads a matrix from
 *   a file. The first two elements of the file are integers
 *   whose values are the dimensions of the matrix ('m' rows
 *   and 'n' columns). What follows are 'm'*'n' values
 *   representing the matrix elements stored in row-major
 *   order.  This function allocates blocks of the matrix to
 *   the MPI processes.
 *
 *   The number of processes must be a square number.
 */
 
void read_checkerboard_matrix (
   char *s,              /* IN - File name */
   void ***subs,         /* OUT - 2D array */
   void **storage,       /* OUT - Array elements */
   MPI_Datatype dtype,   /* IN - Element type */
   int *m,               /* OUT - Array rows */
   int *n,               /* OUT - Array cols */
   MPI_Comm grid_comm)   /* IN - Communicator */
{
   void      *buffer;         /* File buffer */
   int        coords[2];      /* Coords of proc receiving
                                 next row of matrix */
   int        datum_size;     /* Bytes per elements */
   int        dest_id;        /* Rank of receiving proc */
   int        grid_coord[2];  /* Process coords */
   int        grid_id;        /* Process rank */
   int        grid_period[2]; /* Wraparound */
   int        grid_size[2];   /* Dimensions of grid */
   int        i, j, k;
   FILE      *infileptr;      /* Input file pointer */
   void      *laddr;          /* Used when proc 0 gets row */
   int        local_cols;     /* Matrix cols on this proc */
   int        local_rows;     /* Matrix rows on this proc */
   void     **lptr;           /* Pointer into 'subs' */
   int        p;              /* Number of processes */
   void      *raddr;          /* Address of first element
                                 to send */
   void      *rptr;           /* Pointer into 'storage' */
   MPI_Status status;         /* Results of read */

   MPI_Comm_rank (grid_comm, &grid_id);
   MPI_Comm_size (grid_comm, &p);
   datum_size = get_size (dtype);

   /* Process 0 opens file, gets number of rows and
      number of cols, and broadcasts this information
      to the other processes. */

   if (grid_id == 0) {
      infileptr = fopen (s, "r");
      if (infileptr == NULL) *m = 0;
      else {
         fread (m, sizeof(int), 1, infileptr);
         fread (n, sizeof(int), 1, infileptr);
      }
   }
   MPI_Bcast (m, 1, MPI_INT, 0, grid_comm);

   if (!(*m)) MPI_Abort (MPI_COMM_WORLD, OPEN_FILE_ERROR);

   MPI_Bcast (n, 1, MPI_INT, 0, grid_comm);

   /* Each process determines the size of the submatrix
      it is responsible for. */

   MPI_Cart_get (grid_comm, 2, grid_size, grid_period,
      grid_coord);
   local_rows = BLOCK_SIZE(grid_coord[0],grid_size[0],*m);
   local_cols = BLOCK_SIZE(grid_coord[1],grid_size[1],*n);

   /* Dynamically allocate two-dimensional matrix 'subs' */

   *storage = my_malloc (grid_id,
      local_rows * local_cols * datum_size);
   *subs = (void **) my_malloc (grid_id,local_rows*PTR_SIZE);
   lptr = (void *) *subs;
   rptr = (void *) *storage;
   for (i = 0; i < local_rows; i++) {
      *(lptr++) = (void *) rptr;
      rptr += local_cols * datum_size;
   }

   /* Grid process 0 reads in the matrix one row at a time
      and distributes each row among the MPI processes. */

   if (grid_id == 0)
      buffer = my_malloc (grid_id, *n * datum_size);

   /* For each row of processes in the process grid... */
   for (i = 0; i < grid_size[0]; i++) {
      coords[0] = i;

      /* For each matrix row controlled by this proc row...*/
      for (j = 0; j < BLOCK_SIZE(i,grid_size[0],*m); j++) {

         /* Read in a row of the matrix */

         if (grid_id == 0) {
            fread (buffer, datum_size, *n, infileptr);
         }

         /* Distribute it among process in the grid row */

         for (k = 0; k < grid_size[1]; k++) {
            coords[1] = k;

            /* Find address of first element to send */
            raddr = buffer +
               BLOCK_LOW(k,grid_size[1],*n) * datum_size;

            /* Determine the grid ID of the process getting
               the subrow */
            MPI_Cart_rank (grid_comm, coords, &dest_id);

            /* Process 0 is responsible for sending...*/
            if (grid_id == 0) {

               /* It is sending (copying) to itself */
               if (dest_id == 0) {
                  laddr = (*subs)[j];
                  memcpy (laddr, raddr,
                     local_cols * datum_size);

               /* It is sending to another process */
               } else {
                  MPI_Send (raddr,
                     BLOCK_SIZE(k,grid_size[1],*n), dtype,
                  dest_id, 0, grid_comm);
               }

            /* Process 'dest_id' is responsible for
               receiving... */
            } else if (grid_id == dest_id) {
               MPI_Recv ((*subs)[j], local_cols, dtype, 0,
                  0, grid_comm,&status);
            }
         }
      }
   }
   if (grid_id == 0) free (buffer);
}


/*
 *   Function 'read_col_striped_matrix' reads a matrix from a
 *   file.  The first two elements of the file are integers
 *   whose values are the dimensions of the matrix ('m' rows
 *   and 'n' columns).  What follows are 'm'*'n' values
 *   representing the matrix elements stored in row-major
 *   order.  This function allocates blocks of columns of the
 *   matrix to the MPI processes.
 */

void read_col_striped_matrix (
      char         *s,       /* IN - File name */
      void      ***subs,     /* OUT - 2-D array */
      void       **storage,  /* OUT - Array elements */
      MPI_Datatype dtype,    /* IN - Element type */
      int         *m,        /* OUT - Rows */
      int         *n,        /* OUT - Cols */
      MPI_Comm     comm)     /* IN - Communicator */
{
   void      *buffer;        /* File buffer */
   int        datum_size;    /* Size of matrix element */
   int        i, j;
   int        id;            /* Process rank */
   FILE      *infileptr;     /* Input file ptr */
   int        local_cols;    /* Cols on this process */
   void     **lptr;          /* Pointer into 'subs' */
   void      *rptr;          /* Pointer into 'storage' */
   int        p;             /* Number of processes */
   int       *send_count;    /* Each proc's count */
   int       *send_disp;     /* Each proc's displacement */

   MPI_Comm_size (comm, &p);
   MPI_Comm_rank (comm, &id);
   datum_size = get_size (dtype);

   /* Process p-1 opens file, gets number of rows and
      cols, and broadcasts this info to other procs. */

   if (id == (p-1)) {
      infileptr = fopen (s, "r");
      if (infileptr == NULL) *m = 0;
      else {
         fread (m, sizeof(int), 1, infileptr);
         fread (n, sizeof(int), 1, infileptr);
      }
   }
   MPI_Bcast (m, 1, MPI_INT, p-1, comm);

   if (!(*m)) MPI_Abort (comm, OPEN_FILE_ERROR);

   MPI_Bcast (n, 1, MPI_INT, p-1, comm);

   local_cols = BLOCK_SIZE(id,p,*n);

   /* Dynamically allocate two-dimensional matrix 'subs' */

   *storage = my_malloc (id, *m * local_cols * datum_size);
   *subs = (void **) my_malloc (id, *m * PTR_SIZE);
   lptr = (void *) *subs;
   rptr = (void *) *storage;
   for (i = 0; i < *m; i++) {
      *(lptr++) = (void *) rptr;
      rptr += local_cols * datum_size;
   }

   /* Process p-1 reads in the matrix one row at a time and
      distributes each row among the MPI processes. */

   if (id == (p-1))
      buffer = my_malloc (id, *n * datum_size);
   create_mixed_xfer_arrays (id,p,*n,&send_count,&send_disp);
   for (i = 0; i < *m; i++) {
      if (id == (p-1))
         fread (buffer, datum_size, *n, infileptr);
      MPI_Scatterv (buffer, send_count, send_disp, dtype,
         (*storage)+i*local_cols*datum_size, local_cols,
         dtype, p-1, comm);
   }
   free (send_count);
   free (send_disp);
   if (id == (p-1)) free (buffer);
}


/*
 *   Process p-1 opens a file and inputs a two-dimensional
 *   matrix, reading and distributing blocks of rows to the
 *   other processes.
 */

void read_row_striped_matrix (
   char        *s,        /* IN - File name */
   void      ***subs,     /* OUT - 2D submatrix indices */
   void       **storage,  /* OUT - Submatrix stored here */
   MPI_Datatype dtype,    /* IN - Matrix element type */
   int         *m,        /* OUT - Matrix rows */
   int         *n,        /* OUT - Matrix cols */
   MPI_Comm     comm)     /* IN - Communicator */
{
   int          datum_size;   /* Size of matrix element */
   int          i;
   int          id;           /* Process rank */
   FILE        *infileptr;    /* Input file pointer */
   int          local_rows;   /* Rows on this proc */
   void       **lptr;         /* Pointer into 'subs' */
   int          p;            /* Number of processes */
   void        *rptr;         /* Pointer into 'storage' */
   MPI_Status   status;       /* Result of receive */
   int          x;            /* Result of read */

   MPI_Comm_size (comm, &p);
   MPI_Comm_rank (comm, &id);
   datum_size = get_size (dtype);

   /* Process p-1 opens file, reads size of matrix,
      and broadcasts matrix dimensions to other procs */

   if (id == (p-1)) {
      infileptr = fopen (s, "r");
      if (infileptr == NULL) *m = 0;
      else {
         fread (m, sizeof(int), 1, infileptr);
         fread (n, sizeof(int), 1, infileptr);
      }      
   }
   MPI_Bcast (m, 1, MPI_INT, p-1, comm);

   if (!(*m)) MPI_Abort (MPI_COMM_WORLD, OPEN_FILE_ERROR);

   MPI_Bcast (n, 1, MPI_INT, p-1, comm);

   local_rows = BLOCK_SIZE(id,p,*m);
   /* Dynamically allocate matrix. Allow double subscripting
      through 'a'. */

   *storage = (void *) my_malloc (id,
       local_rows * *n * datum_size);
   *subs = (void **) my_malloc (id, local_rows * PTR_SIZE);

   lptr = (void *) &(*subs[0]);
   rptr = (void *) *storage;
   for (i = 0; i < local_rows; i++) {
      *(lptr++)= (void *) rptr;
      rptr += *n * datum_size;
   }

   /* Process p-1 reads blocks of rows from file and
      sends each block to the correct destination process.
      The last block it keeps. */

   if (id == (p-1)) {
      for (i = 0; i < p-1; i++) {
         x = fread (*storage, datum_size,
            BLOCK_SIZE(i,p,*m) * *n, infileptr);
         MPI_Send (*storage, BLOCK_SIZE(i,p,*m) * *n, dtype,
            i, DATA_MSG, comm);
      }
      x = fread (*storage, datum_size, local_rows * *n,
         infileptr);
      fclose (infileptr);
   } else
      MPI_Recv (*storage, local_rows * *n, dtype, p-1,
         DATA_MSG, comm, &status);
}


/*
 *   Open a file containing a vector, read its contents,
 *   and distributed the elements by block among the
 *   processes in a communicator.
 */

void read_block_vector (
    char        *s,      /* IN - File name */
    void       **v,      /* OUT - Subvector */
    MPI_Datatype dtype,  /* IN - Element type */
    int         *n,      /* OUT - Vector length */
    MPI_Comm     comm)   /* IN - Communicator */
{
   int        datum_size;   /* Bytes per element */
   int        i;
   FILE      *infileptr;    /* Input file pointer */
   int        local_els;    /* Elements on this proc */
   MPI_Status status;       /* Result of receive */
   int        id;           /* Process rank */
   int        p;            /* Number of processes */
   int        x;            /* Result of read */

   datum_size = get_size (dtype);
   MPI_Comm_size(comm, &p);
   MPI_Comm_rank(comm, &id);

   /* Process p-1 opens file, determines number of vector
      elements, and broadcasts this value to the other
      processes. */

   if (id == (p-1)) {
      infileptr = fopen (s, "r");
      if (infileptr == NULL) *n = 0;
      else fread (n, sizeof(int), 1, infileptr);
   }
   MPI_Bcast (n, 1, MPI_INT, p-1, comm);
   if (! *n) {
      if (!id) {
         printf ("Input file '%s' cannot be opened\n", s);
         fflush (stdout);
      }
   }

   /* Block mapping of vector elements to processes */

   local_els = BLOCK_SIZE(id,p,*n);

   /* Dynamically allocate vector. */

   *v = my_malloc (id, local_els * datum_size);
   if (id == (p-1)) {
      for (i = 0; i < p-1; i++) {
         x = fread (*v, datum_size, BLOCK_SIZE(i,p,*n),
            infileptr);
         MPI_Send (*v, BLOCK_SIZE(i,p,*n), dtype, i, DATA_MSG,
            comm);
      }
      x = fread (*v, datum_size, BLOCK_SIZE(id,p,*n),
             infileptr);
      fclose (infileptr);
   } else {
      MPI_Recv (*v, BLOCK_SIZE(id,p,*n), dtype, p-1, DATA_MSG,
         comm, &status);
   }
}


/*   Open a file containing a vector, read its contents,
     and replicate the vector among all processes in a
     communicator. */

void read_replicated_vector (
   char        *s,      /* IN - File name */
   void       **v,      /* OUT - Vector */
   MPI_Datatype dtype,  /* IN - Vector type */
   int         *n,      /* OUT - Vector length */
   MPI_Comm     comm)   /* IN - Communicator */
{
   int        datum_size; /* Bytes per vector element */
   int        i;
   int        id;         /* Process rank */
   FILE      *infileptr;  /* Input file pointer */
   int        p;          /* Number of processes */

   MPI_Comm_rank (comm, &id);
   MPI_Comm_size (comm, &p);
   datum_size = get_size (dtype);
   if (id == (p-1)) {
      infileptr = fopen (s, "r");
      if (infileptr == NULL) *n = 0;
      else fread (n, sizeof(int), 1, infileptr);
   }
   MPI_Bcast (n, 1, MPI_INT, p-1, MPI_COMM_WORLD);
   if (! *n) terminate (id, "Cannot open vector file");

   *v = my_malloc (id, *n * datum_size);

   if (id == (p-1)) {
      fread (*v, datum_size, *n, infileptr);
      fclose (infileptr);
   }
   MPI_Bcast (*v, *n, dtype, p-1, MPI_COMM_WORLD);
}

/******************** OUTPUT FUNCTIONS ********************/

/*
 *   Print elements of a doubly-subscripted array.
 */
void print_submatrix_dot (
                          void       **a,       /* OUT - Doubly-subscripted array */
                          MPI_Datatype dtype,   /* OUT - Type of array elements */
                          int          rows,    /* OUT - Matrix rows */
                          int          cols)    /* OUT - Matrix cols */
{
    int i, j;
    
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (dtype == MPI_DOUBLE)
                printf ("%6.3f ", ((double **)a)[i][j]);
            else {
                if (dtype == MPI_FLOAT)
                    printf ("%6.3f ", ((float **)a)[i][j]);
                else if (dtype == MPI_INT){
                    if(((int **)a)[i][j]==0)
                        printf ("%2c ", ' ');
                    else
                        printf ("%2c ", '*');
                }
                else if (dtype == MPI_CHAR){
                    printf ("%c ",((char **)a)[i][j]);
                }
            }
        }
        putchar ('\n');
    }
}
void print_submatrix (
   void       **a,       /* OUT - Doubly-subscripted array */
   MPI_Datatype dtype,   /* OUT - Type of array elements */
   int          rows,    /* OUT - Matrix rows */
   int          cols)    /* OUT - Matrix cols */
{
   int i, j;

   for (i = 0; i < rows; i++) {
      for (j = 0; j < cols; j++) {
         if (dtype == MPI_DOUBLE)
            printf ("%6.3f ", ((double **)a)[i][j]);
         else {
            if (dtype == MPI_FLOAT)
               printf ("%6.3f ", ((float **)a)[i][j]);
            else if (dtype == MPI_INT)
               printf ("%6d ", ((int **)a)[i][j]);
            else if (dtype == MPI_CHAR){
                printf ("%c ",((char **)a)[i][j] );
            }
         }
      }
      putchar ('\n');
   }
}


/*
 *   Print elements of a singly-subscripted array.
 */

void print_subvector (
   void        *a,       /* IN - Array pointer */
   MPI_Datatype dtype,   /* IN - Array type */
   int          n)       /* IN - Array size */
{
   int i;

   for (i = 0; i < n; i++) {
      if (dtype == MPI_DOUBLE)
         printf ("%6.3f ", ((double *)a)[i]);
      else {
         if (dtype == MPI_FLOAT)
            printf ("%6.3f ", ((float *)a)[i]);
         else if (dtype == MPI_INT)
            printf ("%6d ", ((int *)a)[i]);
      }
   }
}


/*
 *   Print a matrix distributed checkerboard fashion among
 *   the processes in a communicator.
 */

void print_checkerboard_matrix (
   void       **a,            /* IN -2D matrix */
   MPI_Datatype dtype,        /* IN -Matrix element type */
   int          m,            /* IN -Matrix rows */
   int          n,            /* IN -Matrix columns */
   MPI_Comm     grid_comm)    /* IN - Communicator */
{
   void      *buffer;         /* Room to hold 1 matrix row */
   int        coords[2];      /* Grid coords of process
                                 sending elements */
   int        datum_size;     /* Bytes per matrix element */
   int        els;            /* Elements received */
   int        grid_coords[2]; /* Coords of this process */
   int        grid_id;        /* Process rank in grid */
   int        grid_period[2]; /* Wraparound */
   int        grid_size[2];   /* Dims of process grid */
   int        i, j, k;
   void      *laddr;          /* Where to put subrow */
   int        local_cols;     /* Matrix cols on this proc */
   int        p;              /* Number of processes */
   int        src;            /* ID of proc with subrow */
   MPI_Status status;         /* Result of receive */

   MPI_Comm_rank (grid_comm, &grid_id);
   MPI_Comm_size (grid_comm, &p);
   datum_size = get_size (dtype);

   MPI_Cart_get (grid_comm, 2, grid_size, grid_period,
      grid_coords);
   local_cols = BLOCK_SIZE(grid_coords[1],grid_size[1],n);

   if (!grid_id)
      buffer = my_malloc (grid_id, n*datum_size);

   /* For each row of the process grid */
   for (i = 0; i < grid_size[0]; i++) {
      coords[0] = i;

      /* For each matrix row controlled by the process row */
      for (j = 0; j < BLOCK_SIZE(i,grid_size[0],m); j++) {

         /* Collect the matrix row on grid process 0 and
            print it */
         if (!grid_id) {
            for (k = 0; k < grid_size[1]; k++) {
               coords[1] = k;
               MPI_Cart_rank (grid_comm, coords, &src);
               els = BLOCK_SIZE(k,grid_size[1],n);
               laddr = buffer +
                  BLOCK_LOW(k,grid_size[1],n) * datum_size;
               if (src == 0) {
                  memcpy (laddr, a[j], els * datum_size);
               } else {
                  MPI_Recv(laddr, els, dtype, src, 0,
                     grid_comm, &status);
               }
            }
            print_subvector (buffer, dtype, n);
            putchar ('\n');
         } else if (grid_coords[0] == i) {
            MPI_Send (a[j], local_cols, dtype, 0, 0,
               grid_comm);
         }
      }
   }
   if (!grid_id) {
      free (buffer);
      putchar ('\n');
   }
}


/*
 *   Print a matrix that has a columnwise-block-striped data
 *   decomposition among the elements of a communicator.
 */

void print_col_striped_matrix (
   void       **a,       /* IN - 2D array */
   MPI_Datatype dtype,   /* IN - Type of matrix elements */
   int          m,       /* IN - Matrix rows */
   int          n,       /* IN - Matrix cols */
   MPI_Comm     comm)    /* IN - Communicator */
{
   MPI_Status status;     /* Result of receive */
   int        datum_size; /* Bytes per matrix element */
   void      *buffer;     /* Enough room to hold 1 row */
   int        i, j;
   int        id;         /* Process rank */
   int        p;          /* Number of processes */
   int*       rec_count;  /* Elements received per proc */
   int*       rec_disp;   /* Offset of each proc's block */

   MPI_Comm_rank (comm, &id);
   MPI_Comm_size (comm, &p);
   datum_size = get_size (dtype);
   create_mixed_xfer_arrays (id, p, n, &rec_count,&rec_disp);

   if (!id)
      buffer = my_malloc (id, n*datum_size);

   for (i = 0; i < m; i++) {
      MPI_Gatherv (a[i], BLOCK_SIZE(id,p,n), dtype, buffer,
         rec_count, rec_disp, dtype, 0, MPI_COMM_WORLD);
      if (!id) {
         print_subvector (buffer, dtype, n);
         putchar ('\n');
      }
   }
   free (rec_count);
   free (rec_disp);
   if (!id) {
      free (buffer);
      putchar ('\n');
   }
}


/*
 *   Print a matrix that is distributed in row-striped
 *   fashion among the processes in a communicator.
 */

void print_row_striped_matrix (
   void **a,            /* IN - 2D array */
   MPI_Datatype dtype,  /* IN - Matrix element type */
   int m,               /* IN - Matrix rows */
   int n,               /* IN - Matrix cols */
   MPI_Comm comm)       /* IN - Communicator */
{
   MPI_Status  status;          /* Result of receive */
   void       *bstorage = NULL;        /* Elements received from
                                   another process */
   void      **b = NULL;               /* 2D array indexing into
                                   'bstorage' */
   int         datum_size;      /* Bytes per element */
   int         i;
   int         id;              /* Process rank */
   int         local_rows;      /* This proc's rows */
   int         max_block_size;  /* Most matrix rows held by
                                   any process */
   int         prompt ;          /* Dummy variable */
   int         p;               /* Number of processes */

   MPI_Comm_rank (comm, &id);
   MPI_Comm_size (comm, &p);
   local_rows = BLOCK_SIZE(id,p,m);
   if (!id) {
      print_submatrix (a, dtype, local_rows, n);
      if (p > 1) {
         datum_size = get_size (dtype);
         max_block_size = BLOCK_SIZE(p-1,p,m);
         bstorage = my_malloc (id,
            max_block_size * n * datum_size);
         b = (void **) my_malloc (id,
            max_block_size * datum_size);
         b[0] = bstorage;
          
         for (i = 1; i < max_block_size; i++) {
            b[i] = b[i-1] + n * datum_size;
         }

         for (i = 1; i < p; i++) {
            MPI_Send (&prompt, 1, MPI_INT, i, PROMPT_MSG,
               MPI_COMM_WORLD);
            MPI_Recv (bstorage, BLOCK_SIZE(i,p,m)*n, dtype,
              i, RESPONSE_MSG, MPI_COMM_WORLD, &status);
           print_submatrix (b, dtype, BLOCK_SIZE(i,p,m), n);
         }
         free (b);
         free (bstorage);
      }
      putchar ('\n');
   } else {
      MPI_Recv (&prompt, 1, MPI_INT, 0, PROMPT_MSG,
         MPI_COMM_WORLD, &status);
      MPI_Send (*a, local_rows * n, dtype, 0, RESPONSE_MSG,
                MPI_COMM_WORLD);
   }
}


/*
 *   Print a vector that is block distributed among the
 *   processes in a communicator.
 */

void print_block_vector (
   void        *v,       /* IN - Address of vector */
   MPI_Datatype dtype,   /* IN - Vector element type */
   int          n,       /* IN - Elements in vector */
   MPI_Comm     comm)    /* IN - Communicator */
{
   int        datum_size; /* Bytes per vector element */
   int        i;
   int        prompt;     /* Dummy variable */
   MPI_Status status;     /* Result of receive */
   void       *tmp;       /* Other process's subvector */
   int        id;         /* Process rank */
   int        p;          /* Number of processes */

   MPI_Comm_size (comm, &p);
   MPI_Comm_rank (comm, &id);
   datum_size = get_size (dtype);

   if (!id) {
      print_subvector (v, dtype, BLOCK_SIZE(id,p,n));
      if (p > 1) {
         tmp = my_malloc (id,BLOCK_SIZE(p-1,p,n)*datum_size);
         for (i = 1; i < p; i++) {
            MPI_Send (&prompt, 1, MPI_INT, i, PROMPT_MSG,
               comm);
            MPI_Recv (tmp, BLOCK_SIZE(i,p,n), dtype, i,
               RESPONSE_MSG, comm, &status);
            print_subvector (tmp, dtype, BLOCK_SIZE(i,p,n));
         }
         free (tmp);
      }
      printf ("\n\n");
   } else {
      MPI_Recv (&prompt, 1, MPI_INT, 0, PROMPT_MSG, comm,
         &status);
      MPI_Send (v, BLOCK_SIZE(id,p,n), dtype, 0,
         RESPONSE_MSG, comm);
   }
}


/*
 *   Print a vector that is replicated among the processes
 *   in a communicator.
 */

void print_replicated_vector (
   void        *v,      /* IN - Address of vector */
   MPI_Datatype dtype,  /* IN - Vector element type */
   int          n,      /* IN - Elements in vector */
   MPI_Comm     comm)   /* IN - Communicator */
{
   int id;              /* Process rank */

   MPI_Comm_rank (comm, &id);
   
   if (!id) {
      print_subvector (v, dtype, n);
      printf ("\n\n");
   }
}

void print_row_striped_matrix_dot (
                               void **a,            /* IN - 2D array */
                               MPI_Datatype dtype,  /* IN - Matrix element type */
                               int m,               /* IN - Matrix rows */
                               int n,               /* IN - Matrix cols */
                               MPI_Comm comm)       /* IN - Communicator */
{
    MPI_Status  status;          /* Result of receive */
    void       *bstorage = NULL;        /* Elements received from
                                         another process */
    void      **b = NULL;               /* 2D array indexing into
                                         'bstorage' */
    int         datum_size;      /* Bytes per element */
    int         i;
    int         id;              /* Process rank */
    int         local_rows;      /* This proc's rows */
    int         max_block_size;  /* Most matrix rows held by
                                  any process */
    int         prompt ;          /* Dummy variable */
    int         p;               /* Number of processes */
    
    MPI_Comm_rank (comm, &id);
    MPI_Comm_size (comm, &p);
    local_rows = BLOCK_SIZE(id,p,m);
    if (!id) {
        print_submatrix_dot (a, dtype, local_rows, n);
        if (p > 1) {
            datum_size = get_size (dtype);
            max_block_size = BLOCK_SIZE(p-1,p,m);
            bstorage = my_malloc (id,
                                  max_block_size * n * datum_size);
            b = (void **) my_malloc (id,
                                     max_block_size * datum_size);
            b[0] = bstorage;
            
            for (i = 1; i < max_block_size; i++) {
                b[i] = b[i-1] + n * datum_size;
            }
            
            for (i = 1; i < p; i++) {
                MPI_Send (&prompt, 1, MPI_INT, i, PROMPT_MSG,
                          MPI_COMM_WORLD);
                MPI_Recv (bstorage, BLOCK_SIZE(i,p,m)*n, dtype,
                          i, RESPONSE_MSG, MPI_COMM_WORLD, &status);
                print_submatrix_dot (b, dtype, BLOCK_SIZE(i,p,m), n);
            }
            free (b);
            free (bstorage);
        }
        putchar ('\n');
    } else {
        MPI_Recv (&prompt, 1, MPI_INT, 0, PROMPT_MSG,
                  MPI_COMM_WORLD, &status);
        MPI_Send (*a, local_rows * n, dtype, 0, RESPONSE_MSG,
                  MPI_COMM_WORLD);
    }
}

void row_transfer(int n, int size, int top_process , int down_process , int* top_array , int* down_array,int** data,MPI_Status status) {
    
    //Send the top array to be the top_array of the top_process.
    if(top_process != -1) {
        MPI_Send(data[0], n, MPI_INT, top_process, 0, MPI_COMM_WORLD);
    }
    //Send the down array to be the top_array of the down_process.
    if(down_process != -1) {
        MPI_Send(data[0]+n*(size-1), n, MPI_INT, down_process, 0, MPI_COMM_WORLD);
    }
    //Receive the top_array from the top process.
    if(top_process != -1) {
        MPI_Recv(top_array, n, MPI_INT, top_process, 0, MPI_COMM_WORLD, &status);
    }
    //Receive the down_array from the down process.
    if(down_process != -1) {
        MPI_Recv(down_array, n, MPI_INT, down_process, 0, MPI_COMM_WORLD, &status);
    }
}

void update(int n, int size, int top_process,  int down_process,  int* top_array,  int* down_array, int** a) {
    
    int tmp_size = size+2;
    int tmp_num = n+2;
     int* tmp_matrix = malloc((size+2)*(n+2)*sizeof(int));
    for(int i=0 ; i<tmp_size ; i++) {
        
        tmp_matrix[i*tmp_num + 0] = 0;
        tmp_matrix[i*tmp_num + tmp_num-1] = 0;
        
        if( i==0 && top_process != -1) {
            for(int j=1 ; j<tmp_num-1 ; j++) {
                tmp_matrix[i*tmp_num + j] = top_array[j-1];
            }
        }
        else if( i==tmp_size-1 && down_process != -1) {
            for(int j=1 ; j<tmp_num-1 ; j++) {
                tmp_matrix[i*tmp_num + j] = down_array[j-1];
            }
        }
        else if (0 < i && i<tmp_size-1) {
            for(int j=1 ; j<tmp_num-1 ; j++) {
                tmp_matrix[i*tmp_num + j] = a[0][(i-1)*n + j-1];
            }
        }
        else {
            for(int j=1 ; j<tmp_num-1 ; j++) {
                tmp_matrix[i*tmp_num + j] = 0;
            }
        }
        
    }
    
    for(int i=1 ; i<=size ; i++) {
        for(int j=1 ; j<=n ; j++) {
            int neighbor = 0;
            neighbor = tmp_matrix[i*tmp_num+j-1] + tmp_matrix[i*tmp_num+j+1]
            + tmp_matrix[(i-1)*tmp_num+j-1]
            + tmp_matrix[(i-1)*tmp_num+j]
            + tmp_matrix[(i-1)*tmp_num+j+1]
            + tmp_matrix[(i+1)*tmp_num+j-1]
            + tmp_matrix[(i+1)*tmp_num+j]
            + tmp_matrix[(i+1)*tmp_num+j+1];
            //if the cell is alive without 2 or 3 neighbors, dead.
            if(a[0][(i-1)*n + j-1] == 1 && neighbor != 2 && neighbor != 3)
                a[0][(i-1)*n + j-1] = 0;
            //if the cell is dead with exactly 3 neighbors, alive.
            if(a[0][(i-1)*n + j-1] == 0 && neighbor ==3)
                a[0][(i-1)*n + j-1] = 1;
        }
    }
    free(tmp_matrix);
}

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
