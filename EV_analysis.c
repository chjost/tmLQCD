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
#include <unistd.h>
#include "getopt.h"
#include "git_hash.h"
//#include <gsl/gsl_math.h>
//#include <gsl/gsl_eigen.h>
//#include <gsl/gsl_complex.h>
//#include <gsl/gsl_complex_math.h>
//#include <gsl/gsl_sort_vector.h>

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
#include "linalg/convert_eo_to_lexic.h"

#define DEBUG 1
#define INVERTER "./invert"
#define REMOVESOURCES 1 // remove all output except for the perambulators
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

inline void spinor_times_su3vec(_Complex double *result, spinor factor1,
    su3_vector factor2) {
  result[0] += factor1.s0.c0 * factor2.c0 + factor1.s0.c1 * factor2.c1
      + factor1.s0.c2 * factor2.c2;
  result[1] += factor1.s1.c0 * factor2.c0 + factor1.s1.c1 * factor2.c1
      + factor1.s1.c2 * factor2.c2;
  result[2] += factor1.s2.c0 * factor2.c0 + factor1.s2.c1 * factor2.c1
      + factor1.s2.c2 * factor2.c2;
  result[3] += factor1.s3.c0 * factor2.c0 + factor1.s3.c1 * factor2.c1
      + factor1.s3.c2 * factor2.c2;
}

int generate_eigensystem();
//int eigensystem_gsl();
int create_invert_sources();
void create_input_files(int const dirac, int const timeslice);
void create_perambulators();

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

  /* Read the input file */
  read_input(input_filename);

  tmlqcd_mpi_init(argc, argv);

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
  if (even_odd_flag) {
    j = init_spinor_field(VOLUMEPLUSRAND / 2, NO_OF_SPINORFIELDS);
  } else {
    j = init_spinor_field(VOLUMEPLUSRAND, NO_OF_SPINORFIELDS);
  }
  if (j != 0) {
    fprintf(stderr, "Not enough memory for spinor fields! Aborting...\n");
    exit(-1);
  }
  init_operators();

  /* define the boundary conditions for the fermion fields */
  boundary(g_kappa);

  /* we need to make sure that we don't have even_odd_flag = 1 */
  /* if any of the operators doesn't use it                    */
  /* in this way even/odd can still be used by other operators */
  for (j = 0; j < no_operators; j++)
    if (!operator_list[j].even_odd_flag)
      even_odd_flag = 0;

  printf("# Generating eigensystem\n");
  generate_eigensystem();
  printf("# generating sources\n");
  create_invert_sources();
  printf("# constructing perambulators\n");
  create_perambulators();

  printf("# program finished without problems\n# Clearing memory\n");

#ifdef MPI
  MPI_Finalize();
#endif
  free_gauge_field();
  free_geometry_indices();
  free_jacobi_field();
  free_spinor_field();

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

//void test_ranlux() {
//  double rnd1 = 0., rnd2 = 0.;
//  char filename[100];
//  sprintf(filename, "rng.lime");
//
//  write_ranlux(filename, 0);
//  ranlxd(&rnd1, 1);
//  //random_seed;
//  //rlxd_level;
//  read_ranlux(filename, 0);
//  ranlxd(&rnd2, 1);
//
//  if (fabs(rnd1 - rnd2) < 1e-6) {
//    fprintf(stdout, "random numbers are the same.\n");
//  } else {
//    fprintf(stderr, "random numbers are not the same.\n");
//  }
//
//  return;
//}
/*
 * create new sources
 */
int create_invert_sources() {
  // read in eigenvectors and save as blocks
  char filename[200];
  char call[150];
  int tslice = 0, vec = 0, point = 0, iter = 0, nop = 0, j = 0;
  int status = 0, t = 0, block = LX * LY * LZ;
  int count = 0;
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *even = NULL, *odd = NULL;
  spinor *dirac0 = NULL;
  spinor *dirac1 = NULL;
  spinor *dirac2 = NULL;
  spinor *dirac3 = NULL;
  FILE * file = NULL;
  WRITER* writer = NULL;

  eigenvector = (su3_vector*) calloc(block, sizeof(su3_vector));

//  allocate all spinor fields
  tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
#if (defined SSE || defined SSE2 || defined SSE3)
  dirac0 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
  dirac0 = tmp;
#endif
  tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
#if (defined SSE || defined SSE2 || defined SSE3)
  dirac1 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
  dirac1 = tmp;
#endif
  tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
#if (defined SSE || defined SSE2 || defined SSE3)
  dirac2 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
  dirac2 = tmp;
#endif
  tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
#if (defined SSE || defined SSE2 || defined SSE3)
  dirac3 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
  dirac3 = tmp;
#endif
  tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
#if (defined SSE || defined SSE2 || defined SSE3)
  even = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
  even = tmp;
#endif
  tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
#if (defined SSE || defined SSE2 || defined SSE3)
  odd = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
  odd = tmp;
#endif

  for (tslice = 0; tslice < T; tslice++) {
    for (vec = 0; vec < no_eigenvalues; vec++) {
      sprintf(filename, "./eigenvector.%03d.%03d.%04d", vec, tslice, nstore);
      read_su3_vector(eigenvector, filename, 0, tslice, 1);

      sprintf(filename, "./b_eigenvector.%03d.%03d.%04d", vec, tslice, nstore);
//      printf("writing eigenvector %s\n", filename);
      if ((file = fopen(filename, "wb")) == NULL ) {
        fprintf(stderr, "could not open eigenvector file %s.\nAborting...\n",
            filename);
        exit(-1);
      }
      count = fwrite(eigenvector, sizeof(su3_vector), block, file);
      if (count != block) {
        fprintf(stderr, "could not write all data to file %s.\n", filename);
      }
      fflush(file);
      fclose(file);

      for (t = 0; t < T; t++) {
        for (point = 0; point < block; point++) {
          // Set the spinor to 0
          _spinor_null(dirac0[block*t + point]);
          _spinor_null(dirac1[block*t + point]);
          _spinor_null(dirac2[block*t + point]);
          _spinor_null(dirac3[block*t + point]);

          // Assign the eigenvector to the correct part of the spinor
          if (t == tslice) {
            _vector_assign(dirac0[block*t + point].s0, eigenvector[point]);
            _vector_assign(dirac1[block*t + point].s1, eigenvector[point]);
            _vector_assign(dirac2[block*t + point].s2, eigenvector[point]);
            _vector_assign(dirac3[block*t + point].s3, eigenvector[point]);
          }
        }
      }

//      write spinor field with entries at dirac 0
      convert_lexic_to_eo(even, odd, dirac0);
      sprintf(filename, "%s.%04d.%02d.%02d", "source0", nstore, tslice, vec);
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &even, &odd, 1, 64);
      destruct_writer(writer);
//      write spinor field with entries at dirac 1
      convert_lexic_to_eo(even, odd, dirac1);
      sprintf(filename, "%s.%04d.%02d.%02d", "source1", nstore, tslice, vec);
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &even, &odd, 1, 64);
      destruct_writer(writer);
      //      write spinor field with entries at dirac 2
      convert_lexic_to_eo(even, odd, dirac2);
      sprintf(filename, "%s.%04d.%02d.%02d", "source2", nstore, tslice, vec);
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &even, &odd, 1, 64);
      destruct_writer(writer);
      //      write spinor field with entries at dirac 3
      convert_lexic_to_eo(even, odd, dirac3);
      sprintf(filename, "%s.%04d.%02d.%02d", "source3", nstore, tslice, vec);
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &even, &odd, 1, 64);
      destruct_writer(writer);
      fflush(stdout);

    }
    create_input_files(4, tslice);
    fflush(stdout);
    for (j = 0; j < 4; j++) {
      sprintf(call, "%s -f dirac%d-cg.input", INVERTER, j);
      printf("trying: %s\n", call);
      fflush(stdout);
      system(call);
    }
  }

//  free(eigenvector);
//  free(dirac0);
//  free(dirac1);
//  free(dirac2);
//  free(dirac3);
//  free(even);
//  free(odd);

  return (0);
}

/*
 * create the perambulators
 */
void create_perambulators() {
  int tsource, tsink, diracindex;
  int nvsource, ndsource, nvsink, ndsink, i, point1, count;
  int blocklength = no_eigenvalues * 4, blocksize = blocklength * blocklength;
  int timeblock = LX * LY * LZ;
  spinor *inverted, *even, *odd, *tmp;
  su3_vector* eigenvector;
  char eigenvectorfile[200], invertedfile[200], perambulatorfile[200];
  READER *reader;
  FILE *file = NULL;
  _Complex double *block;

  tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
#if (defined SSE || defined SSE2 || defined SSE3)
  inverted = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
  inverted = tmp;
#endif
  tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
#if (defined SSE || defined SSE2 || defined SSE3)
  even = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
  even = tmp;
#endif
  tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
#if (defined SSE || defined SSE2 || defined SSE3)
  odd = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
  odd = tmp;
#endif

  eigenvector = (su3_vector*) calloc(timeblock, sizeof(su3_vector));

  block = (_Complex double*) calloc(blocksize, sizeof(_Complex double));
  if (block == NULL ) {
    fprintf(stderr, "not enough space to create perambulator.\nAborting...\n");
    return;
  }
  // iterate through the blocks of the perambulator
  for (tsource = 0; tsource < T; tsource++) {
    for (tsink = 0; tsink < T; tsink++) {
      // set the entries of the block to one
      for (i = 0; i < blocksize; i++) {
        block[0] = 0.0;
      }

      // iterate through the "propagator"
      for (nvsource = 0; nvsource < no_eigenvalues; nvsource++) {
        for (ndsource = 0; ndsource < 4; ndsource++) {
          sprintf(invertedfile, "source%d.%04d.%02d.%02d.inverted", ndsource,
              nstore, tsource, nvsource);
//          printf("reading spinor %s\n", invertedfile);
          read_spinor(even, odd, invertedfile, 0);
          convert_eo_to_lexic(inverted, even, odd);

          // iterate through the eigenvectors
          for (nvsink = 0; nvsink < no_eigenvalues; nvsink++) {
            sprintf(eigenvectorfile, "eigenvector.%03d.%03d.%04d", nvsink,
                tsink, nstore);
//            printf("reading   eigenvector %s\n", eigenvectorfile);
            read_su3_vector(eigenvector, eigenvectorfile, 0, tsink, 1);
            for (point1 = 0; point1 < timeblock; point1++) {
              spinor_times_su3vec(
                  &(block[blocklength * (nvsource * 4 + ndsource) + nvsink * 4]),
                  inverted[point1], eigenvector[point1]);
            }

          } // iterate through the eigenvectors
        }
      } // iterate through the "propagator"
      sprintf(perambulatorfile, "perambulator.%03d.%03d.%04d", tsource, tsink,
          nstore);
//      printf("writing perambulator %s\n", perambulatorfile);
      if ((file = fopen(perambulatorfile, "wb")) == NULL ) {
        fprintf(stderr, "could not open perambulator file %s.\nAborting...\n",
            perambulatorfile);
        exit(-1);
      }
//      printf("writing file %s\n", perambulatorfile);
      count = fwrite(block, sizeof(_Complex double), blocksize, file);
      if (count != blocksize) {
        fprintf(stderr, "could not write all data to file %s.\n",
            perambulatorfile);
      }
      fflush(file);
      fclose(file);
    } // iteration through the blocks of the perambulator
  }
  if (REMOVESOURCES) {
    printf("removing sources\n");
    system("rm source* eigenv* dirac*.input");
  }

  return;
}

//int eigensystem_gsl() {
//  int i, k;
//  FILE* file = NULL;
//
//  gsl_vector * vector_ev = gsl_vector_calloc(192);
//  gsl_matrix_complex * matrix_op = gsl_matrix_complex_calloc(192, 192);
//  gsl_matrix_complex * matrix_ev = gsl_matrix_complex_calloc(192, 192);
//  gsl_eigen_hermv_workspace * workspace_ev = gsl_eigen_hermv_alloc(192);
//
////		unit_g_gauge_field();
//
//// construct operator according to paper 1104.3870v1, eq. (5)
//  gsl_complex compl, compl1; // GSL complex number
//// coordinates of point y and neighbors in configuration:
//  int ix, ix_x, ix_y, ix_z, ix_xd, ix_yd, ix_zd;
//
//  int x2 = 0, y2 = 0, z2 = 0; // coordinates of point y
//  double *tmp_entry1u, *tmp_entry2u, *tmp_entry3u; // su3 entries of the "upper" neighbors
//  double *tmp_entry1d, *tmp_entry2d, *tmp_entry3d; // su3 entries of the "lower" neighbors
//
//  for (x2 = 0; x2 < LX; x2++) {
//    for (y2 = 0; y2 < LY; y2++) {
//      for (z2 = 0; z2 < LZ; z2++) {
//        // get point y and neighbors
//        ix = g_ipt[0][x2][y2][z2];
//        ix_x = g_iup[ix][1]; // +x-direction
//        ix_y = g_iup[ix][2]; // +y-direction
//        ix_z = g_iup[ix][3]; // +z-direction
//        ix_xd = g_idn[ix][1]; // -x-direction
//        ix_yd = g_idn[ix][2]; // -y-direction
//        ix_zd = g_idn[ix][3]; // -z-direction
//
//        GSL_SET_COMPLEX(&compl, -6.0, 0.0);
//        GSL_SET_COMPLEX(&compl1, 1.0, 0.0);
//        tmp_entry1u = (double *) &(g_gauge_field[ix][1]);
//        tmp_entry2u = (double *) &(g_gauge_field[ix][2]);
//        tmp_entry3u = (double *) &(g_gauge_field[ix][3]);
//
//        tmp_entry1d = (double *) &(g_gauge_field[ix_xd][1]);
//        tmp_entry2d = (double *) &(g_gauge_field[ix_yd][2]);
//        tmp_entry3d = (double *) &(g_gauge_field[ix_zd][3]);
//        for (i = 0; i < 3; i++) {
//          gsl_matrix_complex_set(matrix_op, 3 * ix + i, 3 * ix + i, compl);
//        }
//        for (i = 0; i < 3; i++) {
//          for (k = 0; k < 3; k++) {
//
//            GSL_SET_COMPLEX(&compl, *(tmp_entry1u+6*i+2*k),
//                *(tmp_entry1u+6*i+2*k+1));
//            gsl_matrix_complex_set(matrix_op, 3 * ix + k, 3 * ix_x + i, compl);
//            GSL_SET_COMPLEX(&compl, *(tmp_entry2u+6*i+2*k),
//                *(tmp_entry2u+6*i+2*k+1));
//            gsl_matrix_complex_set(matrix_op, 3 * ix + k, 3 * ix_y + i, compl);
//            GSL_SET_COMPLEX(&compl, *(tmp_entry3u+6*i+2*k),
//                *(tmp_entry3u+6*i+2*k+1));
//            gsl_matrix_complex_set(matrix_op, 3 * ix + k, 3 * ix_z + i, compl);
//
//            GSL_SET_COMPLEX(&compl, *(tmp_entry1u+6*i+2*k),
//                -*(tmp_entry1u+6*i+2*k+1));
//            gsl_matrix_complex_set(matrix_op, 3 * ix_x + i, 3 * ix + k, compl);
//            GSL_SET_COMPLEX(&compl, *(tmp_entry2u+6*i+2*k),
//                -*(tmp_entry2u+6*i+2*k+1));
//            gsl_matrix_complex_set(matrix_op, 3 * ix_y + i, 3 * ix + k, compl);
//            GSL_SET_COMPLEX(&compl, *(tmp_entry3u+6*i+2*k),
//                -*(tmp_entry3u+6*i+2*k+1));
//            gsl_matrix_complex_set(matrix_op, 3 * ix_z + i, 3 * ix + k, compl);
//
//            GSL_SET_COMPLEX(&compl, *(tmp_entry1d+6*i+2*k),
//                -*(tmp_entry1d+6*i+2*k+1));
//            gsl_matrix_complex_set(matrix_op, 3 * ix + i, 3 * ix_xd + k, compl);
//            GSL_SET_COMPLEX(&compl, *(tmp_entry2d+6*i+2*k),
//                -*(tmp_entry2d+6*i+2*k+1));
//            gsl_matrix_complex_set(matrix_op, 3 * ix + i, 3 * ix_yd + k, compl);
//            GSL_SET_COMPLEX(&compl, *(tmp_entry3d+6*i+2*k),
//                -*(tmp_entry3d+6*i+2*k+1));
//            gsl_matrix_complex_set(matrix_op, 3 * ix + i, 3 * ix_zd + k, compl);
//
//            GSL_SET_COMPLEX(&compl, *(tmp_entry1d+6*i+2*k),
//                *(tmp_entry1d+6*i+2*k+1));
//            gsl_matrix_complex_set(matrix_op, 3 * ix_xd + k, 3 * ix + i, compl);
//            GSL_SET_COMPLEX(&compl, *(tmp_entry2d+6*i+2*k),
//                *(tmp_entry2d+6*i+2*k+1));
//            gsl_matrix_complex_set(matrix_op, 3 * ix_yd + k, 3 * ix + i, compl);
//            GSL_SET_COMPLEX(&compl, *(tmp_entry3d+6*i+2*k),
//                *(tmp_entry3d+6*i+2*k+1));
//            gsl_matrix_complex_set(matrix_op, 3 * ix_zd + k, 3 * ix + i, compl);
//          }
//        }
//      }
//    }
//  }
//
//  for (k = 0; k < 192; k++) {
//    for (i = k; k < 192; k++) {
//      if ((gsl_matrix_complex_get(matrix_op, i, k).dat[0])
//          != (gsl_matrix_complex_get(matrix_op, k, i).dat[0])) {
//        printf("re not hermitian: i = %i, k = %i\n", i, k);
//      }
//      if ((gsl_matrix_complex_get(matrix_op, i, k).dat[1])
//          != -(gsl_matrix_complex_get(matrix_op, k, i).dat[1])) {
//        printf("im not hermitian: i = %i, k = %i\n", i, k);
//      }
//    }
//  }
//// construction of the lower triangle of the matrix
//  for (k = 0; k < 192; k++) {
//    for (i = k; i < 192; i++) {
//      compl = gsl_matrix_complex_get(matrix_op, i, k);
//      compl1 = gsl_complex_conjugate(compl);
//      gsl_matrix_complex_set(matrix_op, k, i, compl1);
//    }
//  }
//
//// print the non-zero (real) parts of the operator matrix
//  if (DEBUG) {
//    file = fopen("test.dat", "w");
//    if (file == NULL ) {
//      fprintf(stderr, "Could not open file \"test.dat\"\n");
//    } else {
//      for (i = 0; i < 192; i++) {
//        for (k = 0; k < 192; k++) {
//          if ((gsl_matrix_complex_get(matrix_op, i, k)).dat[0] != 0.0)
//            fprintf(file, "%i %i %lf\n", i, k,
//                (gsl_matrix_complex_get(matrix_op, i, k)).dat[0]);
//        }
//      }
//      fclose(file);
//    }
//  }
//// end construction of the operator
//
//// calculate eigensystem
//  gsl_eigen_hermv(matrix_op, vector_ev, matrix_ev, workspace_ev);
//  gsl_sort_vector(vector_ev);
//  for (i = 0; i < 192; i++) {
//    printf("eigenvalue %i: %f\n", i, gsl_vector_get(vector_ev, i));
//  }
//  gsl_vector_free(vector_ev);
//  gsl_matrix_complex_free(matrix_ev);
//  gsl_matrix_complex_free(matrix_op);
//  return 0;
//}

void create_input_files(int const dirac, int const timeslice) {
  char filename[150];
  FILE* file = NULL;
  int j = 0, n_op = 0;
  for (j = 0; j < dirac; j++) {
    sprintf(filename, "dirac%d-cg.input", j);
    file = fopen(filename, "w");
    if (file == NULL ) {
      fprintf(stderr,
          "could not open file %s in create_input_files.\nAborting...\n",
          filename);
      exit(-1);
    }
    fprintf(file, "# automatic generated file for invert\n");
    fprintf(file, "L=%d\nT=%d\n\n", L, T);
    fprintf(file, "DebugLevel = 0\n");
    fprintf(file, "InitialStoreCounter = %d\n", 1000);
    fprintf(file, "Measurements = %d\n", Nmeas);
    fprintf(file, "2kappamu = %6f\n", 0.177);
    fprintf(file, "kappa = %6f\n", 0.177);
    fprintf(file, "BCAngleT = %f\n", 1.0);
    fprintf(file, "GaugeConfigInputFile = conf\n");
    fprintf(file, "UseEvenOdd = yes\n\n");
    fprintf(file, "SourceType = timeslice\n");
    fprintf(file, "ReadSource = yes\n");
    fprintf(file, "SourceTimeslice = %d\n", timeslice);
    fprintf(file, "SourceFileName = source%d\n", j);
    fprintf(file, "NoSamples = %d\n", 1);
    fprintf(file, "Indices = 0-%d\n\n", no_eigenvalues - 1);
    for (n_op = 0; n_op < no_operators; n_op++) {
      switch (operator_list[n_op].type) {
      case 0:
        fprintf(file, "BeginOperator TMWILSON\n");
        break;
      case 1:
        fprintf(file, "BeginOperator OVERLAP\n");
        break;
      case 2:
        fprintf(file, "BeginOperator WILSON\n");
        break;
      case 3:
        fprintf(file, "BeginOperator DBTMWILSON\n");
        break;
      case 4:
        fprintf(file, "BeginOperator CLOVER\n");
        break;
      case 5:
        fprintf(file, "BeginOperator DBCLOVER\n");
        break;
      }
      fprintf(file, "  2kappamu = %6f\n", operator_list[n_op].mu);
      fprintf(file, "  kappa = %6f\n", operator_list[n_op].kappa);
      fprintf(file, "  UseEvenOdd = %s\n",
          (operator_list[n_op].even_odd_flag) ? "yes" : "no");
      switch (operator_list[n_op].solver) {
      case 0:
        fprintf(file, "  Solver = BICGSTAB\n");
        break;
      case 1:
        fprintf(file, "  Solver = CG\n");
        break;
      case 2:
        fprintf(file, "  Solver = GMRES\n");
        break;
      case 3:
        fprintf(file, "  Solver = CGS\n");
        break;
      case 4:
        fprintf(file, "  Solver = MR\n");
        break;
      case 5:
        fprintf(file, "  Solver = BICGSTABELL\n");
        break;
      case 6:
        fprintf(file, "  Solver = FGMRES\n");
        break;
      case 7:
        fprintf(file, "  Solver = GCR\n");
        break;
      case 8:
        fprintf(file, "  Solver = GMRESDR\n");
        break;
      case 9:
        fprintf(file, "  Solver = PCG\n");
        break;
      case 10:
        fprintf(file, "  Solver = DFLGCR\n");
        break;
      case 11:
        fprintf(file, "  Solver = DFLFGMRES\n");
        break;
      case 12:
        fprintf(file, "  Solver = CGMMS\n");
        break;
      case 13:
        fprintf(file, "  Solver = MIXEDCG\n");
        break;
      default:
        break;
      }
      fprintf(file, "  SolverPrecision = %g\n", operator_list[n_op].eps_sq);
      fprintf(file, "  MaxSolverIterations = %d\n",
          operator_list[n_op].maxiter);
      fprintf(file, "  AddDownPropagator = %s\n",
          operator_list[n_op].DownProp ? "yes" : "no");
      fprintf(file, "EndOperator\n");
    }

    fflush(file);
    fclose(file);
  }
  return;
}
