#include <stdio.h>				// needed for printing
#include <math.h>				// needed for tanh, used in init function
#include "params.h"				// model & simulation parameters

#include "mpi.h"

// Modify the parameters of init function to fit the constructs of divided arrays in each processor
void init(double local_u[][N], double local_v[][N], int start, int row_size){
	double uhi, ulo, vhi, vlo;

    uhi = 0.5; ulo = -0.5; vhi = 0.1; vlo = -0.1;


	for (int i=0; i < row_size; i++){
		for (int j=0; j < N; j++) {
            local_u[i][j] = ulo + (uhi - ulo) * 0.5 * (1.0 + tanh((i + start - N / 2) / 16.0)); // Use start to compensate the offset
            local_v[i][j] = vlo + (vhi - vlo) * 0.5 * (1.0 + tanh((j - N / 2) / 16.0));
        }
	}
}

void dxdt(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]){
	double lapu, lapv;
	int up, down, left, right;
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			if (i == 0){
				down = i;
			}
			else{
				down = i-1;
			}
			if (i == N-1){
				up = i;
			}
			else{
				up = i+1;
			}
			if (j == 0){
				left = j;
			}
			else{
				left = j-1;
			}
			if (j == N-1){
				right = j;
			}
			else{
				right = j+1;
			}
			lapu = u[up][j] + u[down][j] + u[i][left] + u[i][right] + -4.0*u[i][j];
			lapv = v[up][j] + v[down][j] + v[i][left] + v[i][right] + -4.0*v[i][j];
			du[i][j] = DD*lapu + u[i][j]*(1.0 - u[i][j])*(u[i][j]-b) - v[i][j];
			dv[i][j] = d*DD*lapv + c*(a*u[i][j] - v[i][j]);
		}
	}
}

void step(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]){
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			u[i][j] += dt*du[i][j];
			v[i][j] += dt*dv[i][j];
		}
	}
}

double norm(double x[N][N]){
	double nrmx = 0.0;
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			nrmx += x[i][j]*x[i][j];
		}
	}
	return nrmx;
}

int main(int argc, char** argv){

    // Init of MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    if (rank == 0){
        double t = 0.0, nrmu, nrmv;
        double u[N][N], v[N][N], du[N][N], dv[N][N];
        int start, end;
        int portion_size = (N - N % (size - 1)) / (size - 1);
        int last_size = portion_size + N % (size - 1);

        FILE *fptr = fopen("nrms.txt", "w");
        fprintf(fptr, "#t\t\tnrmu\t\tnrmv\n");

        // Divided the u and v into average size(The size for last processor may be more than others)
        for (int i = 1; i < size; i++){
            start = (i - 1) * portion_size;
            if (i != size - 1){
                end = start + portion_size;
            }else{
                end = start + last_size;
            }
            // Use MPI_Send() to send the start index and end index of each divided array
            MPI_Send(&start, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&end, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        }

        // Loop to receive assigned u and v from other processors
        for (int i = 1; i < size; i++) {
            start = (i - 1) * portion_size;
            if (i != size - 1){
                MPI_Recv(&u[start], portion_size * N, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&v[start], portion_size * N, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }else{
                MPI_Recv(&u[start], last_size * N, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&v[start], last_size * N, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }


        // time-loop
        for (int k=0; k < M; k++){
            // track the time
            t = dt*k;
            // evaluate the PDE
            dxdt(du, dv, u, v);
            // update the state variables u,v
            step(du, dv, u, v);
            if (k%m == 0){
                // calculate the norms
                nrmu = norm(u);
                nrmv = norm(v);
                printf("t = %2.1f\tu-norm = %2.5f\tv-norm = %2.5f\n", t, nrmu, nrmv);
                fprintf(fptr, "%f\t%f\t%f\n", t, nrmu, nrmv);
            }
        }
        fclose(fptr);

    }else{
        int start, end;

        // Receive the index from the root processor
        MPI_Recv(&start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&end, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int row_size = end - start;
        double local_u[row_size][N], local_v[row_size][N];

        // initialize the state
        init(local_u, local_v, start, row_size);

        // Send back the u and v value to root processor
        MPI_Send(&local_u, row_size * N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&local_v, row_size * N, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);

    }

	MPI_Finalize();
	return 0;
}