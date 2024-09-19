/******************************************************************************
* FILE: mpi_mm.c
* DESCRIPTION:  
*   MPI Matrix Multiply - C Version
*   In this code, the master task distributes a matrix multiply
*   operation to numtasks-1 worker tasks.
*   NOTE:  C and Fortran versions of this code differ because of the way
*   arrays are stored/passed.  C arrays are row-major order but Fortran
*   arrays are column-major order.
* AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
*   Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
* LAST REVISED: 09/29/2021
******************************************************************************/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int main (int argc, char *argv[])
{
    int sizeOfMatrix;
    if (argc == 2)
    {
        sizeOfMatrix = atoi(argv[1]);
    }
    else
    {
        printf("\n Please provide the size of the matrix");
        return 0;
    }
    int	numtasks, taskid, numworkers, source, dest, mtype, rows,
        averow, extra, offset, i, j, k, rc;
    double	a[sizeOfMatrix][sizeOfMatrix],
        b[sizeOfMatrix][sizeOfMatrix],
        c[sizeOfMatrix][sizeOfMatrix];
    MPI_Status status;
    double worker_receive_time = 0,
           worker_calculation_time = 0,
           worker_send_time = 0;
    double whole_computation_time,
           master_initialization_time,
           master_send_receive_time = 0;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (numtasks < 2) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    numworkers = numtasks-1;

    // Create a new communicator for workers
    MPI_Comm worker_comm;
    MPI_Comm_split(MPI_COMM_WORLD, taskid == MASTER ? MPI_UNDEFINED : 1, taskid, &worker_comm);

    double start_time = MPI_Wtime();

    if (taskid == MASTER)
    {
        printf("mpi_mm has started with %d tasks.\n", numtasks);
        printf("Initializing arrays...\n");

        double master_init_start = MPI_Wtime();

        for (i=0; i<sizeOfMatrix; i++)
            for (j=0; j<sizeOfMatrix; j++) {
                a[i][j] = i+j;
                b[i][j] = i*j;
            }

        master_initialization_time = MPI_Wtime() - master_init_start;

        double master_send_start = MPI_Wtime();

        averow = sizeOfMatrix/numworkers;
        extra = sizeOfMatrix%numworkers;
        offset = 0;
        mtype = FROM_MASTER;
        for (dest=1; dest<=numworkers; dest++)
        {
            rows = (dest <= extra) ? averow+1 : averow;   	
            printf("Sending %d rows to task %d offset=%d\n",rows,dest,offset);
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&a[offset][0], rows*sizeOfMatrix, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&b, sizeOfMatrix*sizeOfMatrix, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            offset = offset + rows;
        }

        mtype = FROM_WORKER;
        for (i=1; i<=numworkers; i++)
        {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset][0], rows*sizeOfMatrix, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
            printf("Received results from task %d\n",source);
        }

        master_send_receive_time = MPI_Wtime() - master_send_start;
    }

    if (taskid > MASTER)
    {
        double receive_start = MPI_Wtime();

        mtype = FROM_MASTER;
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&a, rows*sizeOfMatrix, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&b, sizeOfMatrix*sizeOfMatrix, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

        worker_receive_time = MPI_Wtime() - receive_start;

        double calculation_start = MPI_Wtime();

        for (k=0; k<sizeOfMatrix; k++)
            for (i=0; i<rows; i++)
            {
                c[i][k] = 0.0;
                for (j=0; j<sizeOfMatrix; j++)
                    c[i][k] = c[i][k] + a[i][j] * b[j][k];
            }

        worker_calculation_time = MPI_Wtime() - calculation_start;

        double send_start = MPI_Wtime();

        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&c, rows*sizeOfMatrix, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);

        worker_send_time = MPI_Wtime() - send_start;
    }

    whole_computation_time = MPI_Wtime() - start_time;

    // Perform MPI_Reduce operations
    double worker_receive_time_max, worker_receive_time_min, worker_receive_time_sum;
    double worker_calculation_time_max, worker_calculation_time_min, worker_calculation_time_sum;
    double worker_send_time_max, worker_send_time_min, worker_send_time_sum;

    if (worker_comm != MPI_COMM_NULL) {
        MPI_Reduce(&worker_receive_time, &worker_receive_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, worker_comm);
        MPI_Reduce(&worker_receive_time, &worker_receive_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, worker_comm);
        MPI_Reduce(&worker_receive_time, &worker_receive_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, worker_comm);

        MPI_Reduce(&worker_calculation_time, &worker_calculation_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, worker_comm);
        MPI_Reduce(&worker_calculation_time, &worker_calculation_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, worker_comm);
        MPI_Reduce(&worker_calculation_time, &worker_calculation_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, worker_comm);

        MPI_Reduce(&worker_send_time, &worker_send_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, worker_comm);
        MPI_Reduce(&worker_send_time, &worker_send_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, worker_comm);
        MPI_Reduce(&worker_send_time, &worker_send_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, worker_comm);
    }

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("num_procs", numtasks);
    adiak::value("matrix_size", sizeOfMatrix);
    adiak::value("program_name", "master_worker_matrix_multiplication");
    adiak::value("matrix_datatype_size", sizeof(double));

    if (taskid == MASTER)
    {
        printf("******************************************************\n");
        printf("Master Times:\n");
        printf("Whole Computation Time: %f \n", whole_computation_time);
        printf("Master Initialization Time: %f \n", master_initialization_time);
        printf("Master Send and Receive Time: %f \n", master_send_receive_time);
        printf("\n******************************************************\n");

        adiak::value("MPI_Reduce-whole_computation_time", whole_computation_time);
        adiak::value("MPI_Reduce-master_initialization_time", master_initialization_time);
        adiak::value("MPI_Reduce-master_send_receive_time", master_send_receive_time);

        // Receive worker statistics from the first worker
        MPI_Recv(&worker_receive_time_max, 1, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_receive_time_min, 1, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_receive_time_sum, 1, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_calculation_time_max, 1, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_calculation_time_min, 1, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_calculation_time_sum, 1, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_send_time_max, 1, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_send_time_min, 1, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_send_time_sum, 1, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, &status);

        double worker_receive_time_avg = worker_receive_time_sum / numworkers;
        double worker_calculation_time_avg = worker_calculation_time_sum / numworkers;
        double worker_send_time_avg = worker_send_time_sum / numworkers;

        adiak::value("MPI_Reduce-worker_receive_time_max", worker_receive_time_max);
        adiak::value("MPI_Reduce-worker_receive_time_min", worker_receive_time_min);
        adiak::value("MPI_Reduce-worker_receive_time_avg", worker_receive_time_avg);
        adiak::value("MPI_Reduce-worker_calculation_time_max", worker_calculation_time_max);
        adiak::value("MPI_Reduce-worker_calculation_time_min", worker_calculation_time_min);
        adiak::value("MPI_Reduce-worker_calculation_time_avg", worker_calculation_time_avg);
        adiak::value("MPI_Reduce-worker_send_time_max", worker_send_time_max);
        adiak::value("MPI_Reduce-worker_send_time_min", worker_send_time_min);
        adiak::value("MPI_Reduce-worker_send_time_avg", worker_send_time_avg);

        printf("Worker Times:\n");
        printf("Worker Receive Time - Max: %f, Min: %f, Avg: %f\n", worker_receive_time_max, worker_receive_time_min, worker_receive_time_avg);
        printf("Worker Calculation Time - Max: %f, Min: %f, Avg: %f\n", worker_calculation_time_max, worker_calculation_time_min, worker_calculation_time_avg);
        printf("Worker Send Time - Max: %f, Min: %f, Avg: %f\n", worker_send_time_max, worker_send_time_min, worker_send_time_avg);
        printf("\n******************************************************\n");
    }
    else if (taskid == 1)
    {
        // Send worker statistics to the master
        MPI_Send(&worker_receive_time_max, 1, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&worker_receive_time_min, 1, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&worker_receive_time_sum, 1, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&worker_calculation_time_max, 1, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&worker_calculation_time_min, 1, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&worker_calculation_time_sum, 1, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&worker_send_time_max, 1, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&worker_send_time_min, 1, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&worker_send_time_sum, 1, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
    }

    if (worker_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&worker_comm);
    }

    MPI_Finalize();
    return 0;
}