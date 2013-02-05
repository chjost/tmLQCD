/*
 * Created by Christian Jost, Jan 2013
 */

#define MAIN_PROGRAM

#include"lime.h"
#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <signal.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_sort_vector.h>

#include "global.h"
#include "start.h"
#include "read_input.h"
#include "ranlxd.h"
#include <io/gauge.h>
#include <io/utils.h>
#include <io/params.h>
#include <io/su3_vector.h>
#include <io/spinor.h>
#include "measure_gauge_action.h"
#include "geometry_eo.h"
#include "boundary.h"
#include "operator.h"
#include "mpi_init.h"
#ifdef MPI
#include <mpi.h>
#include "xchange.h"
#endif
#ifdef OMP
# include <omp.h>
# include "init/init_omp_accumulators.h"
#endif
#include "init/init.h"
#include "init/init_gauge_field.h"
#include "init/init_geometry_indices.h"
#include "sighandler.h"
#include "su3.h"

#define DEBUG 1
#define _vector_one(r) \
  (r).c0 = 1.;		\
  (r).c1 = 1.;		\
  (r).c2 = 1.;

int generate_eigensystem();
int eigensystem_gsl();
int calculate();

int main(int argc, char* argv[]) {
	int status = 0;

#ifdef MPI
	MPI_Init(&argc, &argv);
#endif

	/* Read the input file */
	read_input("EV_analysis.input");

	tmlqcd_mpi_init(argc, argv);

	if(g_proc_id==0) {
#ifdef SSE
		printf("# The code was compiled with SSE instructions\n");
#endif
#ifdef SSE2
		printf("# The code was compiled with SSE2 instructions\n");
#endif
#ifdef SSE3
		printf("# The code was compiled with SSE3 instructions\n");
#endif
#ifdef P4
		printf("# The code was compiled for Pentium4\n");
#endif
#ifdef OPTERON
		printf("# The code was compiled for AMD Opteron\n");
#endif
#ifdef _GAUGE_COPY
		printf("# The code was compiled with -D_GAUGE_COPY\n");
#endif
#ifdef BGL
		printf("# The code was compiled for Blue Gene/L\n");
#endif
#ifdef BGP
		printf("# The code was compiled for Blue Gene/P\n");
#endif
#ifdef _USE_HALFSPINOR
		printf("# The code was compiled with -D_USE_HALFSPINOR\n");
#endif
#ifdef _USE_SHMEM
		printf("# the code was compiled with -D_USE_SHMEM\n");
#  ifdef _PERSISTENT
		printf("# the code was compiled for persistent MPI calls (halfspinor only)\n");
#  endif
#endif
#ifdef MPI
#  ifdef _NON_BLOCKING
		printf("# the code was compiled for non-blocking MPI calls (spinor and gauge)\n");
#  endif
#endif
		printf("\n");
		fflush(stdout);
	}

#ifdef OMP
	if(omp_num_threads > 0)
	{
		omp_set_num_threads(omp_num_threads);
	}
	else {
		if( g_proc_id == 0 )
			printf("# No value provided for OmpNumThreads, running in single-threaded mode!\n");

		omp_num_threads = 1;
		omp_set_num_threads(omp_num_threads);
	}

	init_omp_accumulators(omp_num_threads);
#endif

#ifndef WITHLAPH
	printf(" Error: WITHLAPH not defined");
	exit(0);
#endif
#ifdef MPI
#ifndef _INDEX_INDEP_GEOM
	printf(" Error: _INDEX_INDEP_GEOM not defined");
	exit(0);
#endif
#ifndef _USE_TSPLITPAR
	printf(" Error: _USE_TSPLITPAR not defined");
	exit(0);
#endif
#endif
#ifdef FIXEDVOLUME
	printf(" Error: FIXEDVOLUME not allowed");
	exit(0);
#endif
	start_ranlux(1, 123456);
	generate_eigensystem();

	printf("**********************************************************\n");
	printf("checking eigenvector\n");

	su3_vector eigenvector[64];
	for(int i=0; i<64; i++) {
		_vector_null(eigenvector[i]);
	}
//	WRITER *writer;
//	construct_writer(&writer, "test_eigenvector.dat", 0);
//	status = write_su3_vector(writer, eigenvector, 64, 0);

	status = read_su3_vector(eigenvector, "eigenvector.000.000.1000", 0, 0);
	printf("read status = %i\n", status);
	if(status == 0) {
		for(int i=0; i<64; i++) {
			printf("%3i: (%lf+%lf i, ", i, creal(eigenvector[i].c0), cimag(eigenvector[i].c0));
			printf("%lf+%lf i, ", creal(eigenvector[i].c1), cimag(eigenvector[i].c1));
			printf("%lf+%lf i)\n", creal(eigenvector[i].c2), cimag(eigenvector[i].c2));
		}
	}
#ifdef MPI
	MPI_Finalize();
#endif
	return status;
}

int generate_eigensystem() {
	int tslice,j,k;
	char conf_filename[50];

	init_gauge_field(VOLUMEPLUSRAND + g_dbw2rand, 0);
	init_geometry_indices(VOLUMEPLUSRAND + g_dbw2rand);

	if(g_proc_id == 0) {
		fprintf(stdout,"The number of processes is %d \n",g_nproc);
		printf("# The lattice size is %d x %d x %d x %d\n",
				(int)(T*g_nproc_t), (int)(LX*g_nproc_x), (int)(LY*g_nproc_y), (int)(g_nproc_z*LZ));
		printf("# The local lattice size is %d x %d x %d x %d\n",
				(int)(T), (int)(LX), (int)(LY),(int) LZ);
		printf("# Computing LapH eigensystem \n");

		fflush(stdout);
	}

	/* define the geometry */
	geometry();



	/* Read Gauge field */
	sprintf(conf_filename, "%s.%.4d", gauge_input_filename, nstore);
	if (g_cart_id == 0) {
		printf("#\n# Trying to read gauge field from file %s in %s precision.\n",
				conf_filename, (gauge_precision_read_flag == 32 ? "single" : "double"));
		fflush(stdout);
	}
	if( (j = read_gauge_field(conf_filename)) !=0) {
		fprintf(stderr, "Error %d while reading gauge field from %s\n Aborting...\n", j, conf_filename);
		exit(-2);
	}


	if (g_cart_id == 0) {
		printf("# Finished reading gauge field.\n");
		fflush(stdout);
	}

#ifdef MPI
	/*For parallelization: exchange the gaugefield */
	xchange_gauge(g_gauge_field);
#endif

	/* Init Jacobi field */
	init_jacobi_field(SPACEVOLUME+SPACERAND,3);

#ifdef MPI
	{
		/* for debugging in parallel set i_gdb = 0 */
		volatile int i_gdb = 8;
		char hostname[256];
		gethostname(hostname, sizeof(hostname));
		printf("PID %d on %s ready for attach\n", getpid(), hostname);
		fflush(stdout);
		if(g_cart_id == 0){
			while (0 == i_gdb){
				sleep(5);
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
#endif

	for (k=0 ; k<3 ; k++)
		random_jacobi_field(g_jacobi_field[k]);


	/* Compute LapH Eigensystem */

	for(tslice=0; tslice<T; tslice++){
		eigenvalues_Jacobi(&no_eigenvalues,5000, eigenvalue_precision,0,tslice,nstore);
	}
	return(0);
}

void dilution() {

}

/*
 * create perambulators and solve
 */
int calculate() {

	char filename[100];
	sprintf(filename, "rng.lime");
	WRITER * writer;
	construct_writer(writer, filename, 0);
	write_ranlux(writer);
//random_seed;
//rlxd_level;
	read_ranlux("rng.lime", 0);


	return(0);
}

int eigensystem_gsl() {
	int i, k;
	FILE* file = NULL;

	gsl_vector * vector_ev = gsl_vector_calloc(192);
	gsl_matrix_complex * matrix_op = gsl_matrix_complex_calloc(192,192);
	gsl_matrix_complex * matrix_ev = gsl_matrix_complex_calloc(192,192);
	gsl_eigen_hermv_workspace * workspace_ev = gsl_eigen_hermv_alloc(192);

	//		unit_g_gauge_field();

	// construct operator according to paper 1104.3870v1, eq. (5)
	gsl_complex compl, compl1; // GSL complex number
	// coordinates of point y and neighbors in configuration:
	int ix, ix_x, ix_y, ix_z, ix_xd, ix_yd, ix_zd;

	int x2 = 0, y2 = 0, z2 = 0; // coordinates of point y
	double *tmp_entry1u, *tmp_entry2u, *tmp_entry3u;// su3 entries of the "upper" neighbors
	double *tmp_entry1d, *tmp_entry2d, *tmp_entry3d;// su3 entries of the "lower" neighbors

	for(x2=0; x2<LX; x2++) {
		for(y2 = 0; y2<LY; y2++) {
			for(z2 = 0; z2 < LZ; z2++) {
				// get point y and neighbors
				ix = g_ipt[0][x2][y2][z2];
				ix_x = g_iup[ix][1];// +x-direction
				ix_y = g_iup[ix][2];// +y-direction
				ix_z = g_iup[ix][3];// +z-direction
				ix_xd = g_idn[ix][1];// -x-direction
				ix_yd = g_idn[ix][2];// -y-direction
				ix_zd = g_idn[ix][3];// -z-direction

				GSL_SET_COMPLEX(&compl, -6.0, 0.0);
				GSL_SET_COMPLEX(&compl1, 1.0, 0.0);
				tmp_entry1u = (double *) &(g_gauge_field[ix][1]);
				tmp_entry2u = (double *) &(g_gauge_field[ix][2]);
				tmp_entry3u = (double *) &(g_gauge_field[ix][3]);

				tmp_entry1d = (double *) &(g_gauge_field[ix_xd][1]);
				tmp_entry2d = (double *) &(g_gauge_field[ix_yd][2]);
				tmp_entry3d = (double *) &(g_gauge_field[ix_zd][3]);
				for(i = 0; i<3; i++) {
					gsl_matrix_complex_set(matrix_op, 3*ix +i, 3*ix +i, compl);
				}
				for(i = 0; i < 3; i++ ){
					for(k=0; k<3; k++) {

						GSL_SET_COMPLEX(&compl, *(tmp_entry1u+6*i+2*k), *(tmp_entry1u+6*i+2*k+1));
						gsl_matrix_complex_set(matrix_op, 3*ix+k, 3*ix_x+i, compl);
						GSL_SET_COMPLEX(&compl, *(tmp_entry2u+6*i+2*k), *(tmp_entry2u+6*i+2*k+1));
						gsl_matrix_complex_set(matrix_op, 3*ix+k, 3*ix_y+i, compl);
						GSL_SET_COMPLEX(&compl, *(tmp_entry3u+6*i+2*k), *(tmp_entry3u+6*i+2*k+1));
						gsl_matrix_complex_set(matrix_op, 3*ix+k, 3*ix_z+i, compl);

						GSL_SET_COMPLEX(&compl, *(tmp_entry1u+6*i+2*k), -*(tmp_entry1u+6*i+2*k+1));
						gsl_matrix_complex_set(matrix_op, 3*ix_x+i, 3*ix+k, compl);
						GSL_SET_COMPLEX(&compl, *(tmp_entry2u+6*i+2*k), -*(tmp_entry2u+6*i+2*k+1));
						gsl_matrix_complex_set(matrix_op, 3*ix_y+i, 3*ix+k, compl);
						GSL_SET_COMPLEX(&compl, *(tmp_entry3u+6*i+2*k), -*(tmp_entry3u+6*i+2*k+1));
						gsl_matrix_complex_set(matrix_op, 3*ix_z+i, 3*ix+k, compl);

						GSL_SET_COMPLEX(&compl, *(tmp_entry1d+6*i+2*k), -*(tmp_entry1d+6*i+2*k+1));
						gsl_matrix_complex_set(matrix_op, 3*ix+i, 3*ix_xd+k, compl);
						GSL_SET_COMPLEX(&compl, *(tmp_entry2d+6*i+2*k), -*(tmp_entry2d+6*i+2*k+1));
						gsl_matrix_complex_set(matrix_op, 3*ix+i, 3*ix_yd+k, compl);
						GSL_SET_COMPLEX(&compl, *(tmp_entry3d+6*i+2*k), -*(tmp_entry3d+6*i+2*k+1));
						gsl_matrix_complex_set(matrix_op, 3*ix+i, 3*ix_zd+k, compl);

						GSL_SET_COMPLEX(&compl, *(tmp_entry1d+6*i+2*k), *(tmp_entry1d+6*i+2*k+1));
						gsl_matrix_complex_set(matrix_op, 3*ix_xd+k, 3*ix+i, compl);
						GSL_SET_COMPLEX(&compl, *(tmp_entry2d+6*i+2*k), *(tmp_entry2d+6*i+2*k+1));
						gsl_matrix_complex_set(matrix_op, 3*ix_yd+k, 3*ix+i, compl);
						GSL_SET_COMPLEX(&compl, *(tmp_entry3d+6*i+2*k), *(tmp_entry3d+6*i+2*k+1));
						gsl_matrix_complex_set(matrix_op, 3*ix_zd+k, 3*ix+i, compl);
					}
				}
			}
		}
	}

	for(k= 0; k<192; k++) {
		for(i=k; k<192; k++) {
			if((gsl_matrix_complex_get(matrix_op, i, k).dat[0]) != (gsl_matrix_complex_get(matrix_op, k, i).dat[0]) ) {
				printf("re not hermitian: i = %i, k = %i\n", i, k);
			}
			if((gsl_matrix_complex_get(matrix_op, i, k).dat[1]) != -(gsl_matrix_complex_get(matrix_op, k, i).dat[1]) ) {
				printf("im not hermitian: i = %i, k = %i\n", i, k);
			}
		}
	}
	// construction of the lower triangle of the matrix
	for(k=0; k<192; k++) {
		for(i=k; i<192; i++) {
			compl = gsl_matrix_complex_get(matrix_op, i, k);
			compl1 = gsl_complex_conjugate(compl);
			gsl_matrix_complex_set(matrix_op, k, i, compl1);
		}
	}

	// print the non-zero (real) parts of the operator matrix
	if(DEBUG) {
		file = fopen("test.dat", "w");
		if(file == NULL) {
			fprintf(stderr, "Could not open file \"test.dat\"\n");
		} else {
			for(i=0; i<192;i++) {
				for(k=0; k<192; k++) {
					if((gsl_matrix_complex_get(matrix_op, i, k)).dat[0] != 0.0)
						fprintf(file, "%i %i %lf\n", i, k, (gsl_matrix_complex_get(matrix_op, i, k)).dat[0]);
				}
			}
			fclose(file);
		}
	}
	// end construction of the operator

	// calculate eigensystem
	gsl_eigen_hermv(matrix_op, vector_ev, matrix_ev, workspace_ev);
	gsl_sort_vector(vector_ev);
	for(i=0; i<192; i++) {
		printf("eigenvalue %i: %f\n", i, gsl_vector_get(vector_ev, i));
	}
	gsl_vector_free(vector_ev);
	gsl_matrix_complex_free(matrix_ev);
	gsl_matrix_complex_free(matrix_op);
	return 0;
}
