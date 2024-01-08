#pragma once
#define _XOPEN_SOURCE

const int N 	= 128;			// domain size
const int M		= 50000;		// number of time steps
const double a 	= 0.3;			// model parameter a
const double b 	= 0.1;			// model parameter b
const double c 	= 0.01;			// model parameter c
const double d 	= 0.0;			// model parameter d
const double dt	= 0.1;			// time step
const double dx	= 2.0;			// spatial resolution
const double DD = 1.0/(dx*dx);	// diffusion scaling
const int m		= 200;			// Norm calculation period

void init(double local_u[N][N], double local_v[N][N], int start, int row_size);

void step(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]);

void dxdt(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]);

double norm(double x[N][N]);