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
#include "getopt.h"
#include "git_hash.h"
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
#include <io/ranlux.h>
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
#include "init/init_jacobi_field.h"
#include "monomial/monomial.h"
#include "solver/eigenvalues_Jacobi.h"
#include "sighandler.h"
#include "su3.h"
#include "su3spinor.h"

#define DEBUG 1
#define _vector_one(r) \
  (r).c0 = 1.;		\
  (r).c1 = 1.;		\
  (r).c2 = 1.;

void usage() {
  fprintf(stdout, "Program for investigating stochastical LapH smearing\n");
  fprintf(stdout, "Version %s \n\n", PACKAGE_VERSION);
  fprintf(stdout, "Please send bug reports to %s\n", PACKAGE_BUGREPORT);
  fprintf(stdout, "Usage:   EV_analysis [options]\n");
  fprintf(stdout, "Options: [-f input-filename]\n");
  fprintf(stdout, "         [-v] more verbosity\n");
  fprintf(stdout, "         [-h|-?] this help\n");
  fprintf(stdout, "         [-V] print version information and exit\n");
  exit(0);
}

int generate_eigensystem();
int eigensystem_gsl();
int calculate();

int main(int argc, char* argv[]) {
  int status = 0, c, j;
  char datafilename[50];
  char parameterfilename[50];
  char conf_filename[50];
  char * input_filename = NULL;

#ifdef MPI
  MPI_Init(&argc, &argv);
#endif

  // check the arguments
  while ((c = getopt(argc, argv, "h?vVf:")) != -1) {
    switch (c) {
    case 'f':
      input_filename = calloc(200, sizeof(char));
      strncpy(input_filename, optarg, 199);
      break;
    case 'v':
      verbose = 1;
      break;
    case 'V':
      fprintf(stdout, "%s %s\n", PACKAGE_STRING, git_hash);
      exit(0);
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  if (input_filename == NULL ) {
    input_filename = "EV_analysis.input";
  }

#ifdef _KOJAK_INST
#pragma pomp inst init
#pragma pomp inst begin(main)
#endif

  DUM_DERI = 8;
  DUM_MATRIX = DUM_DERI + 5;
#if ((defined BGL && defined XLC) || defined _USE_TSPLITPAR)
  NO_OF_SPINORFIELDS = DUM_MATRIX + 3;
#else
  NO_OF_SPINORFIELDS = DUM_MATRIX + 3;
#endif
  if (g_running_phmc) {
    NO_OF_SPINORFIELDS = DUM_MATRIX + 8;
  }
  /* Read the input file */
  read_input(input_filename);

  tmlqcd_mpi_init(argc, argv);

  // Information about the code
  if (g_proc_id == 0) {
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

  //initialise OMP
#ifdef OMP
  if (omp_num_threads > 0) {
    omp_set_num_threads(omp_num_threads);
  } else {
    if (g_proc_id == 0)
      printf(
          "# No value provided for OmpNumThreads, running in single-threaded mode!\n");

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

  // initialise all needed functions
  start_ranlux(1, random_seed);
  init_gauge_field(VOLUMEPLUSRAND + g_dbw2rand, 0);
  init_geometry_indices(VOLUMEPLUSRAND + g_dbw2rand);
  geometry();
  init_jacobi_field(SPACEVOLUME + SPACERAND, 3);
  init_operators();
  if (no_monomials > 0) {
    if (even_odd_flag) {
      j = init_monomials(VOLUMEPLUSRAND / 2, even_odd_flag);
    } else {
      j = init_monomials(VOLUMEPLUSRAND, even_odd_flag);
    }
    if (j != 0) {
      fprintf(stderr,
          "Not enough memory for monomial pseudo fermion fields! Aborting...\n");
      exit(-1);
    }
  }
  if (even_odd_flag) {
    j = init_spinor_field(VOLUMEPLUSRAND / 2, NO_OF_SPINORFIELDS);
  } else {
    j = init_spinor_field(VOLUMEPLUSRAND, NO_OF_SPINORFIELDS);
  }
  if (j != 0) {
     fprintf(stderr, "Not enough memory for gauge_fields! Aborting...\n");
     exit(-1);
   }
  if (g_running_phmc) {
    j = init_chi_spinor_field(VOLUMEPLUSRAND / 2, 20);
    if (j != 0) {
      fprintf(stderr, "Not enough memory for PHMC Chi fields! Aborting...\n");
      exit(-1);
    }
  }
  /* define the boundary conditions for the fermion fields */
  boundary(g_kappa);
#ifdef _USE_HALFSPINOR
  j = init_dirac_halfspinor();
  if (j != 0) {
    fprintf(stderr, "Not enough memory for halffield! Aborting...\n");
    exit(-1);
  }
  if (g_sloppy_precision_flag == 1) {
    j = init_dirac_halfspinor32();
    if (j != 0)
    {
      fprintf(stderr, "Not enough memory for 32-bit halffield! Aborting...\n");
      exit(-1);
    }
  }
#  if (defined _PERSISTENT)
  if (even_odd_flag)
  init_xchange_halffield();
#  endif
#endif

  /* we need to make sure that we don't have even_odd_flag = 1 */
  /* if any of the operators doesn't use it                    */
  /* in this way even/odd can still be used by other operators */
  for (j = 0; j < no_operators; j++)
    if (!operator_list[j].even_odd_flag)
      even_odd_flag = 0;

  if (g_proc_id == 0) {
    fprintf(stdout, "The number of processes is %d \n", g_nproc);
    printf("# The lattice size is %d x %d x %d x %d\n", (int) (T * g_nproc_t),
        (int) (LX * g_nproc_x), (int) (LY * g_nproc_y), (int) (g_nproc_z * LZ));
    printf("# The local lattice size is %d x %d x %d x %d\n", (int) (T),
        (int) (LX), (int) (LY), (int) LZ);
    printf("# Computing LapH eigensystem \n");

    fflush(stdout);
  }
  generate_eigensystem();
  calculate();

#ifdef MPI
  MPI_Finalize();
#endif
  free_gauge_field();
  free_geometry_indices();
  free_jacobi_field();
  free_monomials();
  free_spinor_field();

#ifdef _KOJAK_INST
#pragma pomp inst end(main)
#endif
  return status;
}

int generate_eigensystem() {
  int tslice, j, k;
  char conf_filename[50];

  /* Read Gauge field */
  sprintf(conf_filename, "%s.%.4d", gauge_input_filename, nstore);
  if (g_cart_id == 0) {
    printf("#\n# Trying to read gauge field from file %s in %s precision.\n",
        conf_filename, (gauge_precision_read_flag == 32 ? "single" : "double"));
    fflush(stdout);
  }
  if ((j = read_gauge_field(conf_filename)) != 0) {
    fprintf(stderr,
        "Error %d while reading gauge field from %s\n Aborting...\n", j,
        conf_filename);
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

#ifdef MPI
  {
    /* for debugging in parallel set i_gdb = 0 */
    volatile int i_gdb = 8;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    if(g_cart_id == 0) {
      while (0 == i_gdb) {
        sleep(5);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
#endif

  for (k = 0; k < 3; k++)
    random_jacobi_field(g_jacobi_field[k]);

  /* Compute LapH Eigensystem */

  for (tslice = 0; tslice < T; tslice++) {
    eigenvalues_Jacobi(&no_eigenvalues, 5000, eigenvalue_precision, 0, tslice,
        nstore);
  }
  return (0);
}

void dilution() {

}

void test_ranlux() {
  double rnd1 = 0., rnd2 = 0.;
  char filename[100];
  sprintf(filename, "rng.lime");

  write_ranlux(filename, 0);
  ranlxd(&rnd1, 1);
  //random_seed;
  //rlxd_level;
  read_ranlux(filename, 0);
  ranlxd(&rnd2, 1);

  if (fabs(rnd1 - rnd2) < 1e-6) {
    fprintf(stdout, "random numbers are the same.\n");
  } else {
    fprintf(stderr, "random numbers are not the same.\n");
  }

  return;
}
/*
 * create perambulators and solve
 */
int calculate() {
  // read in eigenvectors and save as blocks
  char filename[150];
  int tslice = 0, vec = 0, point = 0, iter = 0, nop = 0;
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *dirac0 = NULL, *dirac0_e = NULL, *dirac0_o = NULL;
//  spinor *dirac1 = NULL, *dirac2 = NULL, *dirac3 = NULL;
  spinor *dirac0_prop = NULL, *dirac0_prop_e = NULL, *dirac0_prop_o = NULL;
//  spinor *dirac1_prop = NULL, *dirac2_prop = NULL, *dirac3_prop = NULL;
  FILE * file = NULL;
  WRITER* writer = NULL;
  operator *oper;

  eigenvector = (su3_vector*) calloc(LX * LZ * LY * no_eigenvalues,
      sizeof(su3_vector));
  tmp = (spinor*) calloc(LX * LY * LZ + 1, sizeof(spinor));
#if (defined SSE || defined SSE2 || defined SSE3)
  dirac0 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
  dirac0 = tmp;
#endif
  tmp = (spinor*) calloc(LX * LY * LZ + 1, sizeof(spinor));
#if (defined SSE || defined SSE2 || defined SSE3)
  dirac0_e = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
  dirac0_e = tmp;
#endif
  tmp = (spinor*) calloc(LX * LY * LZ + 1, sizeof(spinor));
#if (defined SSE || defined SSE2 || defined SSE3)
  dirac0_o = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
  dirac0_o = tmp;
#endif
  tmp = (spinor*) calloc(LX * LY * LZ + 1, sizeof(spinor));
#if (defined SSE || defined SSE2 || defined SSE3)
  dirac0_prop = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE)
      & ~ALIGN_BASE);
#else
  dirac0_prop = tmp;
#endif
  tmp = (spinor*) calloc(LX * LY * LZ + 1, sizeof(spinor));
#if (defined SSE || defined SSE2 || defined SSE3)
  dirac0_prop_e = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE)
      & ~ALIGN_BASE);
#else
  dirac0_prop_e = tmp;
#endif
  tmp = (spinor*) calloc(LX * LY * LZ + 1, sizeof(spinor));
#if (defined SSE || defined SSE2 || defined SSE3)
  dirac0_prop_o = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE)
      & ~ALIGN_BASE);
#else
  dirac0_prop_o = tmp;
#endif
  sprintf(stdout, "hier\n");
  fflush(stdout);
  for (tslice = 0; tslice < T; tslice++) {
    sprintf(filename, "eigenvector.all.%.3d.%.4d", tslice, nstore);
    read_su3_vector(&(eigenvector[0]), filename, 0, tslice, no_eigenvalues);
    for (vec = 0; vec < no_eigenvalues; vec++) {
      for (point = 0; point < LX * LY * LZ; point++) {
        // Set the spinor to 0
        _spinor_null(dirac0[point]);

        // Set the first guess spinor to one
        _vector_one(dirac0_prop[point].s0);
        _vector_one(dirac0_prop[point].s1);
        _vector_one(dirac0_prop[point].s2);
        _vector_one(dirac0_prop[point].s3);

        // Assign the eigenvector to the correct part of the spinor
        _vector_assign(dirac0[point].s0, eigenvector[vec*LX*LY*LZ+point]);
      }
      convert_lexic_to_eo(dirac0_e, dirac0_o, dirac0);

      for (nop = 0; nop < no_operators; nop++) {
        oper = &(operator_list[nop]);
        sprintf(stdout, "hier\n");
        fflush(stdout);
        iter = invert_eo(//dirac0_prop_e, dirac0_prop_o, dirac0_e, dirac0_o,
            g_spinor_field[0], g_spinor_field[1], g_spinor_field[2], g_spinor_field[3],
            oper->eps_sq, oper->maxiter, oper->solver, oper->rel_prec, 0,
            oper->even_odd_flag, oper->no_extra_masses, oper->extra_masses,
            oper->id);
      }
    }
  }
  free(dirac0);
  free(dirac0_e);
  free(dirac0_o);
  free(dirac0_prop);
  free(dirac0_prop_e);
  free(dirac0_prop_o);
  return (0);
}

int eigensystem_gsl() {
  int i, k;
  FILE* file = NULL;

  gsl_vector * vector_ev = gsl_vector_calloc(192);
  gsl_matrix_complex * matrix_op = gsl_matrix_complex_calloc(192, 192);
  gsl_matrix_complex * matrix_ev = gsl_matrix_complex_calloc(192, 192);
  gsl_eigen_hermv_workspace * workspace_ev = gsl_eigen_hermv_alloc(192);

//		unit_g_gauge_field();

// construct operator according to paper 1104.3870v1, eq. (5)
  gsl_complex compl, compl1; // GSL complex number
// coordinates of point y and neighbors in configuration:
  int ix, ix_x, ix_y, ix_z, ix_xd, ix_yd, ix_zd;

  int x2 = 0, y2 = 0, z2 = 0; // coordinates of point y
  double *tmp_entry1u, *tmp_entry2u, *tmp_entry3u; // su3 entries of the "upper" neighbors
  double *tmp_entry1d, *tmp_entry2d, *tmp_entry3d; // su3 entries of the "lower" neighbors

  for (x2 = 0; x2 < LX; x2++) {
    for (y2 = 0; y2 < LY; y2++) {
      for (z2 = 0; z2 < LZ; z2++) {
        // get point y and neighbors
        ix = g_ipt[0][x2][y2][z2];
        ix_x = g_iup[ix][1]; // +x-direction
        ix_y = g_iup[ix][2]; // +y-direction
        ix_z = g_iup[ix][3]; // +z-direction
        ix_xd = g_idn[ix][1]; // -x-direction
        ix_yd = g_idn[ix][2]; // -y-direction
        ix_zd = g_idn[ix][3]; // -z-direction

        GSL_SET_COMPLEX(&compl, -6.0, 0.0);
        GSL_SET_COMPLEX(&compl1, 1.0, 0.0);
        tmp_entry1u = (double *) &(g_gauge_field[ix][1]);
        tmp_entry2u = (double *) &(g_gauge_field[ix][2]);
        tmp_entry3u = (double *) &(g_gauge_field[ix][3]);

        tmp_entry1d = (double *) &(g_gauge_field[ix_xd][1]);
        tmp_entry2d = (double *) &(g_gauge_field[ix_yd][2]);
        tmp_entry3d = (double *) &(g_gauge_field[ix_zd][3]);
        for (i = 0; i < 3; i++) {
          gsl_matrix_complex_set(matrix_op, 3 * ix + i, 3 * ix + i, compl);
        }
        for (i = 0; i < 3; i++) {
          for (k = 0; k < 3; k++) {

            GSL_SET_COMPLEX(&compl, *(tmp_entry1u+6*i+2*k),
                *(tmp_entry1u+6*i+2*k+1));
            gsl_matrix_complex_set(matrix_op, 3 * ix + k, 3 * ix_x + i, compl);
            GSL_SET_COMPLEX(&compl, *(tmp_entry2u+6*i+2*k),
                *(tmp_entry2u+6*i+2*k+1));
            gsl_matrix_complex_set(matrix_op, 3 * ix + k, 3 * ix_y + i, compl);
            GSL_SET_COMPLEX(&compl, *(tmp_entry3u+6*i+2*k),
                *(tmp_entry3u+6*i+2*k+1));
            gsl_matrix_complex_set(matrix_op, 3 * ix + k, 3 * ix_z + i, compl);

            GSL_SET_COMPLEX(&compl, *(tmp_entry1u+6*i+2*k),
                -*(tmp_entry1u+6*i+2*k+1));
            gsl_matrix_complex_set(matrix_op, 3 * ix_x + i, 3 * ix + k, compl);
            GSL_SET_COMPLEX(&compl, *(tmp_entry2u+6*i+2*k),
                -*(tmp_entry2u+6*i+2*k+1));
            gsl_matrix_complex_set(matrix_op, 3 * ix_y + i, 3 * ix + k, compl);
            GSL_SET_COMPLEX(&compl, *(tmp_entry3u+6*i+2*k),
                -*(tmp_entry3u+6*i+2*k+1));
            gsl_matrix_complex_set(matrix_op, 3 * ix_z + i, 3 * ix + k, compl);

            GSL_SET_COMPLEX(&compl, *(tmp_entry1d+6*i+2*k),
                -*(tmp_entry1d+6*i+2*k+1));
            gsl_matrix_complex_set(matrix_op, 3 * ix + i, 3 * ix_xd + k, compl);
            GSL_SET_COMPLEX(&compl, *(tmp_entry2d+6*i+2*k),
                -*(tmp_entry2d+6*i+2*k+1));
            gsl_matrix_complex_set(matrix_op, 3 * ix + i, 3 * ix_yd + k, compl);
            GSL_SET_COMPLEX(&compl, *(tmp_entry3d+6*i+2*k),
                -*(tmp_entry3d+6*i+2*k+1));
            gsl_matrix_complex_set(matrix_op, 3 * ix + i, 3 * ix_zd + k, compl);

            GSL_SET_COMPLEX(&compl, *(tmp_entry1d+6*i+2*k),
                *(tmp_entry1d+6*i+2*k+1));
            gsl_matrix_complex_set(matrix_op, 3 * ix_xd + k, 3 * ix + i, compl);
            GSL_SET_COMPLEX(&compl, *(tmp_entry2d+6*i+2*k),
                *(tmp_entry2d+6*i+2*k+1));
            gsl_matrix_complex_set(matrix_op, 3 * ix_yd + k, 3 * ix + i, compl);
            GSL_SET_COMPLEX(&compl, *(tmp_entry3d+6*i+2*k),
                *(tmp_entry3d+6*i+2*k+1));
            gsl_matrix_complex_set(matrix_op, 3 * ix_zd + k, 3 * ix + i, compl);
          }
        }
      }
    }
  }

  for (k = 0; k < 192; k++) {
    for (i = k; k < 192; k++) {
      if ((gsl_matrix_complex_get(matrix_op, i, k).dat[0])
          != (gsl_matrix_complex_get(matrix_op, k, i).dat[0])) {
        printf("re not hermitian: i = %i, k = %i\n", i, k);
      }
      if ((gsl_matrix_complex_get(matrix_op, i, k).dat[1])
          != -(gsl_matrix_complex_get(matrix_op, k, i).dat[1])) {
        printf("im not hermitian: i = %i, k = %i\n", i, k);
      }
    }
  }
// construction of the lower triangle of the matrix
  for (k = 0; k < 192; k++) {
    for (i = k; i < 192; i++) {
      compl = gsl_matrix_complex_get(matrix_op, i, k);
      compl1 = gsl_complex_conjugate(compl);
      gsl_matrix_complex_set(matrix_op, k, i, compl1);
    }
  }

// print the non-zero (real) parts of the operator matrix
  if (DEBUG) {
    file = fopen("test.dat", "w");
    if (file == NULL ) {
      fprintf(stderr, "Could not open file \"test.dat\"\n");
    } else {
      for (i = 0; i < 192; i++) {
        for (k = 0; k < 192; k++) {
          if ((gsl_matrix_complex_get(matrix_op, i, k)).dat[0] != 0.0)
            fprintf(file, "%i %i %lf\n", i, k,
                (gsl_matrix_complex_get(matrix_op, i, k)).dat[0]);
        }
      }
      fclose(file);
    }
  }
// end construction of the operator

// calculate eigensystem
  gsl_eigen_hermv(matrix_op, vector_ev, matrix_ev, workspace_ev);
  gsl_sort_vector(vector_ev);
  for (i = 0; i < 192; i++) {
    printf("eigenvalue %i: %f\n", i, gsl_vector_get(vector_ev, i));
  }
  gsl_vector_free(vector_ev);
  gsl_matrix_complex_free(matrix_ev);
  gsl_matrix_complex_free(matrix_op);
  return 0;
}
