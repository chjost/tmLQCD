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
#include "rnd_gauge_trafo.h"
#include "dilution.h"

#define DEBUG 1
#define INVERTER "./invert"

#define REMOVESOURCES 0 // remove all output except for the perambulators
#define _vector_one(r) \
  (r).c0 = 1.;\
  (r).c1 = 1.;\
  (r).c2 = 1.;
#define _vector_I_one(r) \
  (r).c0 = 1.*I;\
  (r).c1 = 1.*I;\
  (r).c2 = 1.*I;

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

inline void spinor_times_su3vec(_Complex double *result, spinor const factor1,
    su3_vector const factor2, int const blocklength) {
  result[0 * blocklength] += factor1.s0.c0 * conj(factor2.c0)
      + factor1.s0.c1 * conj(factor2.c1) + factor1.s0.c2 * conj(factor2.c2);
  result[1 * blocklength] += factor1.s1.c0 * conj(factor2.c0)
      + factor1.s1.c1 * conj(factor2.c1) + factor1.s1.c2 * conj(factor2.c2);
  result[2 * blocklength] += factor1.s2.c0 * conj(factor2.c0)
      + factor1.s2.c1 * conj(factor2.c1) + factor1.s2.c2 * conj(factor2.c2);
  result[3 * blocklength] += factor1.s3.c0 * conj(factor2.c0)
      + factor1.s3.c1 * conj(factor2.c1) + factor1.s3.c2 * conj(factor2.c2);
}

int generate_eigensystem(int const conf);
//int eigensystem_gsl();
int create_invert_sources(int const conf, int const dilution);
void create_input_files(int const dirac, int const timeslice, int const conf,
    int const dilution);
void create_perambulators(int const conf, int const dilution);
void test_system(int const conf);

int main(int argc, char* argv[]) {
  int status = 0, c, j, conf;
  char * input_filename = NULL;
  char call[200];

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

  // init dilution
//  for (j = 0; j < 1; j++) {
//    add_dilution(D_INTER, D_FULL, D_INTER, 2, -1, 4, -(j + 1) * 111111);
//  }
  g_stochastical_run = 0;
//  add_dilution(D_FULL, D_FULL, D_FULL, 0, 0, 0, 111111);
//  add_dilution(D_NONE, D_FULL, D_FULL, 0, 0, 0, 222222);
//  add_dilution(D_FULL, D_NONE, D_FULL, 0, 0, 0, 333333);
//  add_dilution(D_FULL, D_FULL, D_NONE, 0, 0, 0, 444444);

  /* define the boundary conditions for the fermion fields */
  boundary(g_kappa);

  /* we need to make sure that we don't have even_odd_flag = 1 */
  /* if any of the operators doesn't use it                    */
  /* in this way even/odd can still be used by other operators */
  for (j = 0; j < no_operators; j++)
    if (!operator_list[j].even_odd_flag)
      even_odd_flag = 0;

  // main loop
  for (conf = nstore; conf < nstore + Nmeas; conf += Nsave) {
//    printf("2KappaMu = %e", g_mu);
    printf("\n# Generating eigensystem for conf %d\n", conf);
    fflush(stdout);
    generate_eigensystem(conf);

    if (g_stochastical_run != 0) {
      for (j = 0; j < no_dilution; j++) {
        // restart the RNG
        start_ranlux(1, dilution_list[j].seed);

        //generate the sources
        printf("\n# generating sources (%d of %d)\n", j + 1, no_dilution);
        fflush(stdout);
        create_invert_sources(conf, j);

// construct the perambulators
        printf("\n# constructing perambulators (%d of %d)\n", j + 1,
            no_dilution);
        fflush(stdout);
        create_perambulators(conf, j);

// clean up
        if (REMOVESOURCES && j == (no_dilution - 1)) {
          printf("\n# removing sources\n");
          sprintf(call,
              "rm source?.%04d.* eigenv*.%04d dirac*.input output.para", conf,
              conf);
          system(call);
        }
      }
    } else {
      //generate the sources
      printf("\n# generating sources\n");
      fflush(stdout);
      create_invert_sources(conf, 1);

      // construct the perambulators
      printf("\n# constructing perambulators\n");
      fflush(stdout);
      create_perambulators(conf, 1);

      // clean up
      if (REMOVESOURCES) {
        printf("\n# removing sources\n");
        sprintf(call, "rm source?.%04d.* eigenv*.%04d dirac*.input output.para",
            conf, conf);
        system(call);
      }
    }
    sprintf(call,
        "cp perambulator*.%04d /dsk/lattice02-0/jost/data/8times16_d/", conf);
    system(call);
//    sprintf(call,
//        "cp randomvector*.%04d /dsk/lattice02-0/jost/data/8times16_interlace/",
//        conf);
//    system(call);
    sprintf(call,
        "cp b_eigenvector*.%04d /dsk/lattice02-0/jost/data/8times16_d", conf);
    system(call);
  }

  printf("\n# program finished without problems\n# Clearing memory\n");

#ifdef MPI
  MPI_Finalize();
#endif
  free_gauge_field();
  free_geometry_indices();
  free_jacobi_field();
  free_spinor_field();

  return status;
}

int generate_eigensystem(int const conf) {
  int tslice, j, k;
  char conf_filename[50];

  /* Read Gauge field */
  sprintf(conf_filename, "%s.%.4d", gauge_input_filename, conf);
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
//  unit_g_gauge_field();
//  rnd_gauge_trafo(1, g_gauge_field);

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
        conf);
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
 * create and invert new sources
 */
int create_invert_sources(int const conf, int const dilution) {
  // local variables
  char filename[200];
  char call[150];
  int tslice = 0, dir = 0, vec = 0, point = 0, j = 0;
  int status = 0, t = 0, d = 0, v = 0, block = LX * LY * LZ;
  int interlace_size, t_end = -1, t_step = -1, t_add = T;
  int l_end = -1, l_step = -1, l_add = no_eigenvalues;
  int d_end = -1;
  int count;
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *even = NULL, *odd = NULL;
  spinor *dirac0 = NULL;
  spinor *dirac1 = NULL;
  spinor *dirac2 = NULL;
  spinor *dirac3 = NULL;
  WRITER* writer = NULL;
  double *rnd_vector = NULL;
  FILE* file;
  int index = 0;

  if (g_stochastical_run == 0) {
    t_end = T;
    d_end = 4;
    l_end = no_eigenvalues;
  } else {
    // dilution in time
    if (dilution_list[dilution].type[0] == D_FULL) {
      t_end = T;
      t_step = 1;
    } else if (dilution_list[dilution].type[0] == D_INTER) {
      t_end = dilution_list[dilution].size[0];
      t_step = dilution_list[dilution].size[0];
    } else if (dilution_list[dilution].type[0] == D_BLOCK) {
      t_end = dilution_list[dilution].size[0];
      t_add = dilution_list[dilution].size[0];
      t_step = 1;
    } else if (dilution_list[dilution].type[0] == D_NONE) {
      t_end = 1;
      t_step = 1;
    }
    // dilution in spin space
    if (dilution_list[dilution].type[1] == D_FULL) {
      d_end = no_eigenvalues;
    } else if (dilution_list[dilution].type[1] == D_NONE) {
      d_end = 1;
    }
    //dilution in LapH space
    if (dilution_list[dilution].type[2] == D_FULL) {
      l_end = no_eigenvalues;
      l_step = 1;
    } else if (dilution_list[dilution].type[2] == D_INTER) {
      l_end = dilution_list[dilution].size[2];
      l_step = dilution_list[dilution].size[2];
    } else if (dilution_list[dilution].type[2] == D_BLOCK) {
      l_end = dilution_list[dilution].size[2];
      l_add = dilution_list[dilution].size[2];
      l_step = 1;
    } else if (dilution_list[dilution].type[2] == D_NONE) {
      l_end = 1;
      l_step = 1;
    }
  }

  interlace_size = t_end * d_end * l_end;

  if (g_stochastical_run != 0) {
    rnd_vector = (double*) calloc(interlace_size, sizeof(double));
    ranlxd(rnd_vector, interlace_size);
    sprintf(filename, "randomvector.%03d.%04d", dilution, conf);
    if ((file = fopen(filename, "wb")) == NULL ) {
      fprintf(stderr, "Could not open file %s for random vector\nAborting...\n",
          filename);
      exit(-1);
    }
    count = fwrite(rnd_vector, sizeof(double), interlace_size, file);

    if (count != interlace_size) {
      fprintf(stderr,
          "Could not print all data to file %s for random vector (%d of %d)\nAborting...\n",
          filename, count, interlace_size);
      exit(-1);
    }
    fclose(file);
  }

//  allocate spinors and eigenvectors

  eigenvector = (su3_vector*) calloc(block, sizeof(su3_vector));

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

  for (tslice = 0; tslice < t_end; tslice++) {
    // set the spinors to zero
    zero_spinor_field(dirac0, VOLUMEPLUSRAND);
    zero_spinor_field(dirac1, VOLUMEPLUSRAND);
    zero_spinor_field(dirac2, VOLUMEPLUSRAND);
    zero_spinor_field(dirac3, VOLUMEPLUSRAND);

    for (vec = 0; vec < l_end; vec++) {
      if (g_stochastical_run == 0) {
        sprintf(filename, "./eigenvector.%03d.%03d.%04d", vec, tslice, conf);
        read_su3_vector(eigenvector, filename, 0, tslice, 1);
        for (point = 0; point < block; point++) {
          _vector_assign(dirac0[block*tslice + point].s0, eigenvector[point]);
          _vector_assign(dirac1[block*tslice + point].s1, eigenvector[point]);
          _vector_assign(dirac2[block*tslice + point].s2, eigenvector[point]);
          _vector_assign(dirac3[block*tslice + point].s3, eigenvector[point]);
        }
      } else {
        if (dilution_list[dilution].type[1] == D_NONE) {
          for (t = tslice; t < t_add; t += t_step) {
            for (v = vec; v < no_eigenvalues; v += l_step) {
              sprintf(filename, "./eigenvector.%03d.%03d.%04d", vec, tslice,
                  conf);
              read_su3_vector(eigenvector, filename, 0, tslice, 1);
              index = t * no_eigenvalues * 4 + v * 4;
              for (point = 0; point < block; point++) {
                _vector_add_mul( dirac0[block*tslice + point].s0,
                    rnd_vector[index+0], eigenvector[point]);
                _vector_add_mul( dirac0[block*tslice + point].s1,
                    rnd_vector[index+1], eigenvector[point]);
                _vector_add_mul( dirac0[block*tslice + point].s2,
                    rnd_vector[index+2], eigenvector[point]);
                _vector_add_mul( dirac0[block*tslice + point].s3,
                    rnd_vector[index+3], eigenvector[point]);
              }
            }
          }
        } else if (dilution_list[dilution].type[1] == D_FULL) {
          for (t = tslice; t < t_add; t += t_step) {
            for (v = vec; v < no_eigenvalues; v += l_step) {
              sprintf(filename, "./eigenvector.%03d.%03d.%04d", vec, tslice,
                  conf);
              read_su3_vector(eigenvector, filename, 0, tslice, 1);
              index = t * no_eigenvalues * 4 + v * 4;

              for (point = 0; point < block; point++) {
                _vector_add_mul( dirac0[block*tslice + point].s0,
                    rnd_vector[index+0], eigenvector[point]);
                _vector_add_mul( dirac1[block*tslice + point].s1,
                    rnd_vector[index+1], eigenvector[point]);
                _vector_add_mul( dirac2[block*tslice + point].s2,
                    rnd_vector[index+2], eigenvector[point]);
                _vector_add_mul( dirac3[block*tslice + point].s3,
                    rnd_vector[index+3], eigenvector[point]);
              }
            }
          }
        }
      }

//      write spinor field with entries at dirac 0
      convert_lexic_to_eo(even, odd, dirac0);
      sprintf(filename, "%s.%04d.%02d.%02d", "source0", conf, tslice, vec);
//      printf("writing file %s\n", filename);
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &even, &odd, 1, 64);
      destruct_writer(writer);
      if ((g_stochastical_run == 0)
          || (g_stochastical_run == 1
              && dilution_list[dilution].type[1] == D_FULL)) {
//      write spinor field with entries at dirac 1
        convert_lexic_to_eo(even, odd, dirac1);
        sprintf(filename, "%s.%04d.%02d.%02d", "source1", conf, tslice, vec);
//      printf("writing file %s\n", filename);
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
//      write spinor field with entries at dirac 2
        convert_lexic_to_eo(even, odd, dirac2);
        sprintf(filename, "%s.%04d.%02d.%02d", "source2", conf, tslice, vec);
//      printf("writing file %s\n", filename);
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
//      write spinor field with entries at dirac 3
        convert_lexic_to_eo(even, odd, dirac3);
        sprintf(filename, "%s.%04d.%02d.%02d", "source3", conf, tslice, vec);
//      printf("writing file %s\n", filename);
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        fflush(stdout);
      }

    }
  }

  if (g_stochastical_run == 0) {
    for (tslice = 0; tslice < t_end; tslice++) {
      create_input_files(d_end, tslice, conf, 0);
      for (j = 0; j < d_end; j++) {
        sprintf(call, "%s -f dirac%d-cg.input", INVERTER, j);
        printf("\n\ntrying: %s for conf %4d, t %3d\n", call, conf,
            tslice);
        fflush(stdout);
        system(call);
      }
    }
  } else {
    for (tslice = 0; tslice < t_end; tslice++) {
      create_input_files(d_end, tslice, conf, dilution);
      for (j = 0; j < d_end; j++) {
        sprintf(call, "%s -f dirac%d-cg.input", INVERTER, j);
        printf("\n\ntrying: %s for conf %4d, t %3d, dilution %d\n", call, conf,
            tslice, dilution);
        fflush(stdout);
        system(call);
      }
    }
  }

  free(eigenvector);
  free(dirac0);
  free(dirac1);
  free(dirac2);
  free(dirac3);
  free(even);
  free(odd);
  if(g_stochastical_run != 0) {
    free(rnd_vector);
  }

  return (0);
}

/*
 * create the perambulators
 */
void create_perambulators(int const conf, int const dilution) {
  // local parameters
  int tsource, tsink;
  int nvsource, ndsource, nvsink, i, point1, count;
  int t_end = -1, l_end = -1, d_end = 4;
  int blockwidth, blocksize, blockheigth = 4 * no_eigenvalues;
  int timeblock = LX * LY * LZ;
  spinor *inverted, *even, *odd, *tmp;
  su3_vector* eigenvector;
  char eigenvectorfile[200], invertedfile[200], perambulatorfile[200];
  FILE *file = NULL;
  _Complex double *block;

  // set the correct parameters for the loops
  if (g_stochastical_run == 0) {
    t_end = T;
    d_end = 4;
    l_end = no_eigenvalues;
  } else { // dilution in spin space not implemented!
    if (dilution_list[dilution].type[0] == D_FULL) {
      t_end = T;
    } else if (dilution_list[dilution].type[0] == D_INTER) {
      t_end = dilution_list[dilution].size[0];
    } else if (dilution_list[dilution].type[0] == D_BLOCK) {
      t_end = dilution_list[dilution].size[0];
    } else if (dilution_list[dilution].type[0] == D_NONE) {
      t_end = 1;
    }

    if (dilution_list[dilution].type[1] == D_FULL) {
      d_end = 4;
    } else if (dilution_list[dilution].type[1] == D_NONE) {
      d_end = 1;
    }

    if (dilution_list[dilution].type[2] == D_FULL) {
      l_end = no_eigenvalues;
    } else if (dilution_list[dilution].type[2] == D_INTER) {
      l_end = dilution_list[dilution].size[2];
    } else if (dilution_list[dilution].type[2] == D_BLOCK) {
      l_end = dilution_list[dilution].size[2];
    } else if (dilution_list[dilution].type[2] == D_NONE) {
      l_end = 1;
    }
  }
  blockwidth = d_end * l_end;
  blocksize = blockheigth * blockwidth;

  // allocate the needed memory for spinors, eigenvectors and peramulator
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
  for (tsource = 0; tsource < t_end; tsource++) {
    for (tsink = 0; tsink < T; tsink++) {
      // set the entries of the block to zero
      for (i = 0; i < blocksize; i++) {
        block[i] = 0.0;
      }

      // iterate through the inverted source
      for (nvsource = 0; nvsource < l_end; nvsource++) {
        for (ndsource = 0; ndsource < d_end; ndsource++) {
          sprintf(invertedfile, "source%d.%04d.%02d.%02d.inverted", ndsource,
              conf, tsource, nvsource);
          read_spinor(even, odd, invertedfile, 0);
          convert_eo_to_lexic(inverted, even, odd);

          // iterate through the eigenvectors
          for (nvsink = 0; nvsink < no_eigenvalues; nvsink++) {
            sprintf(eigenvectorfile, "eigenvector.%03d.%03d.%04d", nvsink,
                tsink, conf);
            read_su3_vector(eigenvector, eigenvectorfile, 0, tsink, 1);

            // binary dump of the eigenvectors, needed for the operators
            sprintf(eigenvectorfile, "./b_eigenvector.%03d.%03d.%04d", nvsink,
                tsink, conf);
            if ((file = fopen(eigenvectorfile, "wb")) == NULL ) {
              fprintf(stderr,
                  "could not open eigenvector file %s.\nAborting...\n",
                  eigenvectorfile);
              exit(-1);
            }
            count = fwrite(eigenvector, sizeof(su3_vector), timeblock, file);
            if (count != timeblock) {
              fprintf(stderr, "could not write all data to file %s.\n",
                  eigenvectorfile);
            }
            fclose(file);

            for (point1 = 0; point1 < timeblock; point1++) {
              spinor_times_su3vec(
                  &(block[blockwidth * (nvsink * 4) + nvsource * 4 + ndsource]),
                  inverted[timeblock * tsink + point1], eigenvector[point1],
                  blockwidth);
            }
          } // iterate through the eigenvectors
        }
      } // iterate through the "propagator"

      // save the perambulator
      // naming convention: perambulator[_i(dilution)].(tsource).(tsink).(configuration)
      if (g_stochastical_run == 0) {
        sprintf(perambulatorfile, "perambulator.%03d.%03d.%04d", tsource, tsink,
            conf);
      } else {
        sprintf(perambulatorfile, "perambulator_i.%02d.%03d.%03d.%04d",
            dilution, tsource, tsink, conf);
      }
      if ((file = fopen(perambulatorfile, "wb")) == NULL ) {
        fprintf(stderr, "could not open perambulator file %s.\nAborting...\n",
            perambulatorfile);
        exit(-1);
      }
      count = fwrite(block, sizeof(_Complex double), blocksize, file);
      if (count != blocksize) {
        fprintf(stderr, "could not write all data to file %s.\n",
            perambulatorfile);
      }
      fflush(file);
      fclose(file);

    } // iteration through the blocks of the perambulator
  }

  free(even);
  free(odd);
  free(inverted);
  free(eigenvector);
  free(block);

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

void create_input_files(int const dirac, int const timeslice, int const conf,
    int const dilution) {
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
    fprintf(file, "InitialStoreCounter = %d\n", conf);
    fprintf(file, "Measurements = %d\n", 1);
    fprintf(file, "2kappamu = %f\n", g_mu);
    fprintf(file, "kappa = %f\n", g_kappa);
    fprintf(file, "BCAngleT = %f\n", 1.0);
    fprintf(file, "GaugeConfigInputFile = %s\n", gauge_input_filename);
    fprintf(file, "UseEvenOdd = yes\n\n");
    fprintf(file, "SourceType = timeslice\n");
    fprintf(file, "ReadSource = yes\n");
    fprintf(file, "SourceTimeslice = %d\n", timeslice);
    fprintf(file, "SourceFileName = source%d\n", j);
    fprintf(file, "NoSamples = %d\n", 1);
    if (g_stochastical_run == 0) {
      fprintf(file, "Indices = 0-%d\n\n", no_eigenvalues - 1);
    } else {
      fprintf(file, "Indices = 0-%d\n\n", dilution_list[dilution].size[2] - 1);
//      fprintf(file, "Indices = 0-%d\n\n", no_eigenvalues - 1);
    }
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

void test_system(int const conf) {
  char filename[150];
  spinor *inverted;
  spinor *even;
  spinor *odd;
  spinor *tmp;
  su3_vector *eigenvector;
  int vol = LX * LY * LZ;
  WRITER *writer;
  double one = 1.0;

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

  eigenvector = (su3_vector*) calloc(vol, sizeof(su3_vector));

  for (int i = 0; i < vol; i++) {
    _vector_I_one(eigenvector[i]);
  }
  for (int vec = 0; vec < no_eigenvalues; vec++) {
    for (int t = 0; t < T; t++) {
      sprintf(filename, "eigenvector.%03d.%03d.%04d", vec, t, conf);
      construct_writer(&writer, filename, 0);
      write_su3_vector(writer, &one, eigenvector, 64, t, 1);
      destruct_writer(writer);
    }
  }

  for (int i = 0; i < vol * T; i++) {
    _vector_I_one(inverted[i].s0);
    _vector_I_one(inverted[i].s1);
    _vector_I_one(inverted[i].s2);
    _vector_I_one(inverted[i].s3);
  }
  convert_lexic_to_eo(even, odd, inverted);
  for (int dir = 0; dir < 4; dir++) {
    for (int t = 0; t < T; t++) {
      for (int vec = 0; vec < no_eigenvalues; vec++) {
        sprintf(filename, "source%d.%04d.%02d.%02d.inverted", dir, conf, t,
            vec);
        construct_writer(&writer, filename, 0);
        write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
      }
    }
  }

  free(inverted);
  free(even);
  free(odd);
  free(eigenvector);
  return;
}
