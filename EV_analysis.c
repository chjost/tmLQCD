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
#include <limits.h>
#include "getopt.h"
#include "git_hash.h"

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
#include "dilution.h"
#include "smearing/hex_3d.h"
#include "smearing/utils.h"
#include "buffers/utils.h"

#define BINARYINPUT 1
#define ONESLICE 0
#define DEBUG 1
#define SMEARING 0
#define SMEAR_ITER 2
#define SMEAR_COEFF1 0.76f // should be smaller than 1
#define SMEAR_COEFF2 0.95f // should be smaller than 1
#define INVERTER "./invert"
#define EIGENSYSTEMPATH "../"
#define REMOVESOURCES 0 // remove all output except for the perambulators
#define _vector_one(r) \
  (r).c0 = 1. + 0.*I;\
  (r).c1 = 1. + 0.*I;\
  (r).c2 = 1. + 0.*I;
#define _vector_I_one(r) \
  (r).c0 = 0. + 1.*I;\
  (r).c1 = 0. + 1.*I;\
  (r).c2 = 0. + 1.*I;
#define _vector_const(r, c, d) \
  (r).c0 = 0. + 0.*I;\
  (r).c1 = 0. + 0.*I;\
  (r).c2 = c + d*I;

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

static int read_binary_eigenvector(su3_vector * const s, char * filename) {
  FILE *infile = fopen(filename, "rb");
  if (infile == NULL ) {
    fprintf(stderr, "Unable to find file %s.\nReturning...\n", filename);
    return -1;
  }
  fread(s, sizeof(su3_vector), SPACEVOLUME * no_eigenvalues, infile);
  fclose(infile);

  return 0;
}

inline static void vectorcjg_times_spinor(_Complex double *result,
    su3_vector const factor1, spinor const factor2, int const blocklength) {
  result[0 * blocklength] += conj(factor1.c0) * factor2.s0.c0
      + conj(factor1.c1) * factor2.s0.c1 + conj(factor1.c2) * factor2.s0.c2;
  result[1 * blocklength] += conj(factor1.c0) * factor2.s1.c0
      + conj(factor1.c1) * factor2.s1.c1 + conj(factor1.c2) * factor2.s1.c2;
  result[2 * blocklength] += conj(factor1.c0) * factor2.s2.c0
      + conj(factor1.c1) * factor2.s2.c1 + conj(factor1.c2) * factor2.s2.c2;
  result[3 * blocklength] += conj(factor1.c0) * factor2.s3.c0
      + conj(factor1.c1) * factor2.s3.c1 + conj(factor1.c2) * factor2.s3.c2;
}

inline static void spinor_times_spinor(spinor result, spinor const factor1,
    spinor const factor2) {
  result.s0.c0 += conj(factor1.s0.c0) * factor2.s0.c0;
  result.s0.c1 += conj(factor1.s0.c1) * factor2.s0.c1;
  result.s0.c2 += conj(factor1.s0.c2) * factor2.s0.c2;
  result.s1.c0 += conj(factor1.s1.c0) * factor2.s1.c0;
  result.s1.c1 += conj(factor1.s1.c1) * factor2.s1.c1;
  result.s1.c2 += conj(factor1.s1.c2) * factor2.s1.c2;
  result.s2.c0 += conj(factor1.s2.c0) * factor2.s2.c0;
  result.s2.c1 += conj(factor1.s2.c1) * factor2.s2.c1;
  result.s2.c2 += conj(factor1.s2.c2) * factor2.s2.c2;
  result.s3.c0 += conj(factor1.s3.c0) * factor2.s3.c0;
  result.s3.c1 += conj(factor1.s3.c1) * factor2.s3.c1;
  result.s3.c2 += conj(factor1.s3.c2) * factor2.s3.c2;
}

static void rnd_z2_vector(_Complex double *v, const int N) {
  ranlxd((double*) v, 2 * N);
  for (int i = 0; i < N; ++i) {
    if (creal(v[i]) < 0.5 && cimag(v[i]) < 0.5)
      v[i] = 1 / sqrt(2) + I * 1 / sqrt(2);
    else if (creal(v[i]) >= 0.5 && cimag(v[i]) < 0.5)
      v[i] = -1 / sqrt(2) + I * 1 / sqrt(2);
    else if (creal(v[i]) < 0.5 && cimag(v[i]) >= 0.5)
      v[i] = 1 / sqrt(2) - I * 1 / sqrt(2);
    else
      v[i] = -1 / sqrt(2) - I * 1 / sqrt(2);
  }
  return;
}

int generate_eigensystem(int const conf);
int create_invert_sources(int const conf, int const dilution);
void create_propagators(int const conf, int const dilution);
void create_perambulators(int const conf, int const dilution);
void create_stochastic_perambulators(int const conf, int const dilution);

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
      input_filename = calloc(100, sizeof(char));
      strncpy(input_filename, optarg, 99);
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
    input_filename = calloc(100, sizeof(char));
    strncpy(input_filename, "EV_analysis.input", 99);
  }

  if (access(input_filename, F_OK) == -1) {
    fprintf(stderr,
        "No input file specified and fallback file EV_analysis.input does not exist!\nAborting...\n");
  }

  /* Read the input file */
  read_input(input_filename);
  tmlqcd_mpi_init(argc, argv);

  //initialise OMP
#ifdef OMP
  if (omp_num_threads > 0) {
    omp_set_num_threads(omp_num_threads);
    printf(" threads: %d\n", omp_num_threads);
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
  initialize_gauge_buffers(12);
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
  g_stochastical_run = 1;

// up quarks
//  add_dilution(D_FULL, D_FULL, D_INTER, 0, 0, 8, 3771, D_UP, D_STOCH);
//  add_dilution(D_FULL, D_FULL, D_INTER, 0, 0, 8, 989898, D_UP, D_STOCH);
//  add_dilution(D_INTER, D_FULL, D_INTER, 16, 0, 8, 1227, D_UP, D_STOCH);
//  add_dilution(D_INTER, D_FULL, D_INTER, 16, 0, 8, 1337, D_UP, D_STOCH);
//  add_dilution(D_INTER, D_FULL, D_INTER, 16, 0, 4, 38432, D_UP, D_STOCH);
//  add_dilution(D_INTER, D_FULL, D_INTER, 16, 0, 4, 834234, D_UP, D_STOCH);

// up quarks, interblock in time
//  add_dilution(D_INTERBLOCK, D_FULL, D_INTER, 0, 0, 8, 622331, D_UP, D_STOCH);
//  add_dilution(D_INTERBLOCK, D_FULL, D_INTER, 0, 0, 8, 276960, D_UP, D_STOCH);
//  add_dilution(D_INTERBLOCK, D_FULL, D_INTER, 0, 0, 8, 852000, D_UP, D_STOCH);
//  add_dilution(D_INTERBLOCK, D_FULL, D_INTER, 0, 0, 8, 862587, D_UP, D_STOCH);
//  add_dilution(D_INTERBLOCK, D_FULL, D_INTER, 0, 0, 8, 32377, D_UP, D_STOCH);

// up quarks, one timeslice only
  add_dilution(D_FULL, D_FULL, D_FULL, 0, 0, 8, 1534, D_UP, D_STOCH);
//  add_dilution(D_FULL, D_FULL, D_FULL, 0, 0, 8, 7648956, D_UP, D_STOCH);

// up quarks, stochastische sink
// source
//  add_dilution(D_FULL, D_FULL, D_INTER, 0, 0, 8, 4561, D_UP, D_STOCH);
//  add_dilution(D_FULL, D_FULL, D_INTER, 0, 0, 8, 186512, D_UP, D_STOCH);
//  add_dilution(D_INTER, D_FULL, D_INTER, 16, 0, 8, 1865, D_UP, D_STOCH);
//  add_dilution(D_INTER, D_FULL, D_INTER, 16, 0, 8, 486541, D_UP, D_STOCH);
// sink
//  add_dilution(D_FULL, D_FULL, D_INTER, 0, 0, 8, 4715, D_UP, D_STOCH);
//  add_dilution(D_FULL, D_FULL, D_INTER, 0, 0, 8, 883345, D_UP, D_STOCH);
//  add_dilution(D_INTER, D_FULL, D_INTER, 16, 0, 8, 9152, D_UP, D_STOCH);
//  add_dilution(D_INTER, D_FULL, D_INTER, 16, 0, 8, 743583, D_UP, D_STOCH);

// down quarks, stochastische sink
//source
//  add_dilution(D_FULL, D_FULL, D_INTER, 0, 0, 8, 1584, D_UP, D_STOCH);
//  add_dilution(D_FULL, D_FULL, D_INTER, 0, 0, 8, 684345, D_UP, D_STOCH);
//  add_dilution(D_INTER, D_FULL, D_INTER, 16, 0, 8, 3724, D_UP, D_STOCH);
//  add_dilution(D_INTER, D_FULL, D_INTER, 16, 0, 8, 842357, D_UP, D_STOCH);
//  sink
//  add_dilution(D_FULL, D_FULL, D_INTER, 0, 0, 8, 1857, D_UP, D_STOCH);
//  add_dilution(D_FULL, D_FULL, D_INTER, 0, 0, 8, 284884, D_UP, D_STOCH);
//  add_dilution(D_INTER, D_FULL, D_INTER, 16, 0, 8, 3589, D_UP, D_STOCH);
//  add_dilution(D_INTER, D_FULL, D_INTER, 16, 0, 8, 278625, D_UP, D_STOCH);

// strange quarks
//  add_dilution(D_FULL, D_FULL, D_INTER, 0, 0, 8, 2536, D_STRANGE, D_STOCH);
//  add_dilution(D_FULL, D_FULL, D_INTER, 0, 0, 8, 456354, D_STRANGE, D_STOCH);
//  add_dilution(D_INTER, D_FULL, D_INTER, 16, 0, 8, 9762, D_STRANGE, D_STOCH);
//  add_dilution(D_INTER, D_FULL, D_INTER, 16, 0, 8, 812934, D_STRANGE, D_STOCH);

//  charm quarks
//  add_dilution(D_FULL, D_FULL, D_INTER, 0, 0, 8, 1256, D_CHARM, D_STOCH);
//  add_dilution(D_FULL, D_FULL, D_INTER, 0, 0, 8, 348786, D_CHARM, D_STOCH);
//  add_dilution(D_INTER, D_FULL, D_INTER, 16, 0, 8, 497649, D_CHARM, D_STOCH);
//  add_dilution(D_INTER, D_FULL, D_INTER, 16, 0, 8, 272574, D_CHARM, D_STOCH);

// down quarks
//  add_dilution(D_FULL, D_FULL, D_INTER, 0, 0, 8, 2134, D_DOWN, D_STOCH);
//  add_dilution(D_FULL, D_FULL, D_INTER, 0, 0, 8, 595959, D_DOWN, D_STOCH);
//  add_dilution(D_INTER, D_FULL, D_INTER, 16, 0, 8, 11227, D_DOWN, D_STOCH);
//  add_dilution(D_INTER, D_FULL, D_INTER, 16, 0, 8, 11337, D_DOWN, D_STOCH);

// test cases (time, dirac, laph, int, int, int, seed, up/down, stoch/local)
//  add_dilution(D_FULL, D_FULL, D_FULL, 0, 0, 0, 111111, D_UP, D_STOCH);
//  add_dilution(D_NONE, D_NONE, D_NONE, 0, 0, 8, 3771, D_UP, D_STOCH);

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
    printf("2KappaMu = %e", g_mu);
    printf("\n# Generating eigensystem for conf %d\n", conf);
    fflush(stdout);
    generate_eigensystem(conf);

    if (g_stochastical_run != 0) {
      for (j = 0; j < no_dilution; j++) {
        //generate the sources
        printf("\n# generating sources (%d of %d)\n", j + 1, no_dilution);
        fflush(stdout);
        start_ranlux(1, dilution_list[j].seed ^ conf);
        create_invert_sources(conf, j);

// construct the perambulators
//        printf("\n# constructing perambulators (%d of %d)\n", j + 1,
//            no_dilution);
//        fflush(stdout);
//        create_perambulators(conf, j);

// construct stochastic perambulators
//        printf("\n# constructing stochastic perambulators (%d of %d)\n", j + 1,
//            no_dilution);
//        fflush(stdout);
//        create_stochastic_perambulators(conf, j);

// construct the propagators
//        printf("\n# constructing propagators (%d of %d)\n", j + 1, no_dilution);
//        fflush(stdout);
//        create_propagators(conf, j);

// clean up
        if (REMOVESOURCES) {
          printf("\n# removing sources\n");
          sprintf(call, "rm source?.%04d.* dirac*.input", conf);
          system(call);
          if (j == (no_dilution - 1)) {
            sprintf(call, "rm eigenv*.%04d output.para", conf);
            system(call);
          }
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
//      create_perambulators(conf, 1);

// clean up
      if (REMOVESOURCES) {
        printf("\n# removing sources\n");
        sprintf(call, "rm source?.%04d.* eigenv*.%04d dirac*.input output.para",
            conf, conf);
        system(call);
      }
    }
  }

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

  for (k = 0; k < 3; k++) {
    random_jacobi_field(g_jacobi_field[k], SPACEVOLUME + SPACERAND);
  }

  // smearing
#if SMEARING
  smearing_control_t *mysmearing = construct_smearing_control(HEX_3D, 0,
      SMEAR_ITER, SMEAR_COEFF1, SMEAR_COEFF2);
  smear(mysmearing, _AS_GAUGE_FIELD_T(g_gauge_field) );
  copy_gauge_field(&(_AS_GAUGE_FIELD_T(g_gauge_field) ), mysmearing->result);
#if DEBUG
  if (mysmearing->smearing_performed)
  printf("smearing performed\n");
  fflush(stdout);
#endif
#endif

  /* Compute LapH Eigensystem */
#ifdef OMP
#pragma omp master
  {
#endif
    for (tslice = 0; tslice < T; tslice++) {
      eigenvalues_Jacobi(&no_eigenvalues, 5000, eigenvalue_precision, 0, tslice,
          conf);
    }
#ifdef OMP
  }
#endif

#if SMEARING
  free_smearing_control(mysmearing);
#endif
  return (0);
}

/*
 * create and invert new sources
 */
int create_invert_sources(int const conf, int const dilution) {

// normal run without stochastic part
  if (g_stochastical_run == 0) {
    // local variables
    char filename[200];
    char call[150];
    int tslice = 0, vec = 0, point = 0, j = 0;
    int status = 0, block = LX * LY * LZ;
    su3_vector * eigenvector = NULL;
    spinor *tmp = NULL;
    spinor *even = NULL, *odd = NULL;
    spinor *dirac0 = NULL;
    spinor *dirac1 = NULL;
    spinor *dirac2 = NULL;
    spinor *dirac3 = NULL;
    WRITER* writer = NULL;

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

    for (tslice = 0; tslice < T; tslice++) {
      for (vec = 0; vec < no_eigenvalues; vec++) {
        // zero the spinor fields
        zero_spinor_field(dirac0, VOLUMEPLUSRAND);
        zero_spinor_field(dirac1, VOLUMEPLUSRAND);
        zero_spinor_field(dirac2, VOLUMEPLUSRAND);
        zero_spinor_field(dirac3, VOLUMEPLUSRAND);

        // read in the eigenvector and distribute it to the source
        sprintf(filename, "./eigenvector.%03d.%03d.%04d", vec, tslice, conf);
        read_su3_vector(eigenvector, filename, 0, tslice, 1);
        for (point = 0; point < block; point++) {
          _vector_assign(dirac0[block*tslice + point].s0, eigenvector[point]);
          _vector_assign(dirac1[block*tslice + point].s1, eigenvector[point]);
          _vector_assign(dirac2[block*tslice + point].s2, eigenvector[point]);
          _vector_assign(dirac3[block*tslice + point].s3, eigenvector[point]);
        }

        // write spinor field with entries at dirac 0
        convert_lexic_to_eo(even, odd, dirac0);
        sprintf(filename, "%s.%04d.%02d.%02d", "source0", conf, tslice, vec);
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 1
        convert_lexic_to_eo(even, odd, dirac1);
        sprintf(filename, "%s.%04d.%02d.%02d", "source1", conf, tslice, vec);
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 2
        convert_lexic_to_eo(even, odd, dirac2);
        sprintf(filename, "%s.%04d.%02d.%02d", "source2", conf, tslice, vec);
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 3
        convert_lexic_to_eo(even, odd, dirac3);
        sprintf(filename, "%s.%04d.%02d.%02d", "source3", conf, tslice, vec);
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);

        create_input_files(4, tslice, conf, -1, 0);
        for (j = 0; j < 4; j++) {
          sprintf(call, "%s -f dirac%d-cg.input", INVERTER, j);
          printf("\n\ntrying: %s for conf %d, t %d\n", call, conf, tslice);
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
    // stochastic part
  } else {
    // full spin dilution
    if (dilution_list[dilution].type[1] == D_FULL) {
      // full time dilution
      if (dilution_list[dilution].type[0] == D_FULL) {
        // full LapH dilution
        if (dilution_list[dilution].type[2] == D_FULL) {
#if ONESLICE
          create_source_t1_df_lf(conf, dilution, INVERTER);
#else
          create_source_ti_df_li(conf, dilution, INVERTER);
#endif
          // no LapH dilution
        } else if (dilution_list[dilution].type[2] == D_NONE) {
          create_source_ti_df_li(conf, dilution, INVERTER);

          // interlace LapH dilution
        } else if (dilution_list[dilution].type[2] == D_INTER) {
          create_source_ti_df_li(conf, dilution, INVERTER);

          // block LapH dilution
        } else if (dilution_list[dilution].type[2] == D_BLOCK) {
          create_source_ti_df_lb(conf, dilution, INVERTER);

        } // LapH dilution end

        // no time dilution
      } else if (dilution_list[dilution].type[0] == D_NONE) {
        // full LapH dilution
        if (dilution_list[dilution].type[2] == D_FULL) {
          create_source_ti_df_li(conf, dilution, INVERTER);

          // no LapH dilution
        } else if (dilution_list[dilution].type[2] == D_NONE) {
          create_source_ti_df_li(conf, dilution, INVERTER);

          // interlace LapH dilution
        } else if (dilution_list[dilution].type[2] == D_INTER) {
          create_source_ti_df_li(conf, dilution, INVERTER);

          // block LapH dilution
        } else if (dilution_list[dilution].type[2] == D_BLOCK) {
          create_source_ti_df_lb(conf, dilution, INVERTER);

        } // LapH dilution end

        // interlace time dilution
      } else if (dilution_list[dilution].type[0] == D_INTER) {
        // full LapH dilution
        if (dilution_list[dilution].type[2] == D_FULL) {
          create_source_ti_df_li(conf, dilution, INVERTER);

          // no LapH dilution
        } else if (dilution_list[dilution].type[2] == D_NONE) {
          create_source_ti_df_li(conf, dilution, INVERTER);

          // interlace LapH dilution
        } else if (dilution_list[dilution].type[2] == D_INTER) {
          create_source_ti_df_li(conf, dilution, INVERTER);

          // block LapH dilution
        } else if (dilution_list[dilution].type[2] == D_BLOCK) {
          create_source_ti_df_lb(conf, dilution, INVERTER);

        }

        // block time dilution
      } else if (dilution_list[dilution].type[0] == D_BLOCK) {
        // full LapH dilution
        if (dilution_list[dilution].type[2] == D_FULL) {
          create_source_tb_df_li(conf, dilution, INVERTER);

          // no LapH dilution
        } else if (dilution_list[dilution].type[2] == D_NONE) {
          create_source_tb_df_li(conf, dilution, INVERTER);

          // interlace LapH dilution
        } else if (dilution_list[dilution].type[2] == D_INTER) {
          create_source_tb_df_li(conf, dilution, INVERTER);

          // block LapH dilution
        } else if (dilution_list[dilution].type[2] == D_BLOCK) {
          create_source_tb_df_lb(conf, dilution, INVERTER);

        } // LapH dilution
      } else if (dilution_list[dilution].type[0] == D_INTERBLOCK) {
        printf("entering correct loop\n");
        create_source_tbi2_df_li(conf, dilution, INVERTER);
      } // time dilution

      // no spin dilution
    } else {
      // full time dilution
      if (dilution_list[dilution].type[0] == D_FULL) {
        // full LapH dilution
        if (dilution_list[dilution].type[2] == D_FULL) {
          create_source_ti_dn_li(conf, dilution, INVERTER);

          // no LapH dilution
        } else if (dilution_list[dilution].type[2] == D_NONE) {
          create_source_ti_dn_li(conf, dilution, INVERTER);

          // interlace LapH dilution
        } else if (dilution_list[dilution].type[2] == D_INTER) {
          create_source_ti_dn_li(conf, dilution, INVERTER);

          // block LapH dilution
        } else if (dilution_list[dilution].type[2] == D_BLOCK) {
          create_source_ti_df_lb(conf, dilution, INVERTER);

        } // LapH dilution end

        // no time dilution
      } else if (dilution_list[dilution].type[0] == D_NONE) {
        // full LapH dilution
        if (dilution_list[dilution].type[2] == D_FULL) {
          create_source_ti_dn_li(conf, dilution, INVERTER);

          // no LapH dilution
        } else if (dilution_list[dilution].type[2] == D_NONE) {
          create_source_ti_dn_li(conf, dilution, INVERTER);

          // interlace LapH dilution
        } else if (dilution_list[dilution].type[2] == D_INTER) {
          create_source_ti_dn_li(conf, dilution, INVERTER);

          // block LapH dilution
        } else if (dilution_list[dilution].type[2] == D_BLOCK) {
          create_source_ti_dn_lb(conf, dilution, INVERTER);

        } // LapH dilution end

        // interlace time dilution
      } else if (dilution_list[dilution].type[0] == D_INTER) {
        // full LapH dilution
        if (dilution_list[dilution].type[2] == D_FULL) {
          create_source_ti_dn_li(conf, dilution, INVERTER);

          // no LapH dilution
        } else if (dilution_list[dilution].type[2] == D_NONE) {
          create_source_ti_dn_li(conf, dilution, INVERTER);

          // interlace LapH dilution
        } else if (dilution_list[dilution].type[2] == D_INTER) {
          create_source_ti_dn_li(conf, dilution, INVERTER);

          // block LapH dilution
        } else if (dilution_list[dilution].type[2] == D_BLOCK) {
          create_source_ti_dn_lb(conf, dilution, INVERTER);

        }

        // block time dilution
      } else if (dilution_list[dilution].type[0] == D_BLOCK) {
        // full LapH dilution
        if (dilution_list[dilution].type[2] == D_FULL) {
          create_source_tb_dn_li(conf, dilution, INVERTER);

          // no LapH dilution
        } else if (dilution_list[dilution].type[2] == D_NONE) {
          create_source_tb_dn_li(conf, dilution, INVERTER);

          // interlace LapH dilution
        } else if (dilution_list[dilution].type[2] == D_INTER) {
          create_source_tb_dn_li(conf, dilution, INVERTER);

          // block LapH dilution
        } else if (dilution_list[dilution].type[2] == D_BLOCK) {
          create_source_tb_dn_lb(conf, dilution, INVERTER);

        } // LapH dilution
      } // time dilution
    } // spin dilution end
  } // end stochastic part

  return (0);
}

/*
 * create propagators
 */
void create_propagators(int const conf, int const dilution) {
// set the correct parameters for the loops
  int t_end = -1, l_end = -1, d_end = 4;
  if (g_stochastical_run == 0) {
    t_end = T;
    d_end = 4;
    l_end = no_eigenvalues;
  } else { // dilution in spin space not implemented!
    if (dilution_list[dilution].type[0] == D_FULL) {
      t_end = T;
    } else if (dilution_list[dilution].type[0] == D_INTER
        || dilution_list[dilution].type[0] == D_BLOCK) {
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
    } else if (dilution_list[dilution].type[2] == D_INTER
        || dilution_list[dilution].type[2] == D_BLOCK) {
      l_end = dilution_list[dilution].size[2];
    } else if (dilution_list[dilution].type[2] == D_NONE) {
      l_end = 1;
    }
  }
#if DEBUG
  printf("\nparameters for propagator: t %d, d %d, l %d\n", t_end, d_end,
      l_end);
#endif

#ifdef OMP
#pragma omp parallel \
  shared(t_end, l_end, d_end)
  {
#endif
    int time = 0, count = 0, vec = 0, dirac = 0;
    spinor *inverted = NULL, *even = NULL, *odd = NULL, *tmp = NULL;
    char invertedfile[200], propagatorfile[200];
    FILE *file = NULL;
    // allocate the needed memory for spinors, eigenvectors and peramulator
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    inverted =
        (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    inverted = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(inverted);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    even = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    even = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(inverted);
      free(even);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    odd = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    odd = tmp;
#endif

    // save each propagator into new file
    // iterate over time
#ifdef OMP
#pragma omp for private(time)
#endif
    for (time = 0; time < t_end; time++) {
      // iterate over the LapH space
      for (vec = 0; vec < l_end; vec++) {
        // iterate over the dirac space
        for (dirac = 0; dirac < d_end; dirac++) {

          // read in inverted source
          sprintf(invertedfile, "source%d.%04d.%02d.%02d.inverted", dirac, conf,
              time, vec);
#if DEBUG
#ifdef OMP
          printf("thread %d reading file %s\n", omp_get_thread_num(),
              invertedfile);
#else
          printf("reading file %s\n", invertedfile);
#endif
#endif
          read_spinor(even, odd, invertedfile, 0);
          convert_eo_to_lexic(inverted, even, odd);

          // save to file
          if (g_stochastical_run == 0) {
            sprintf(propagatorfile, "propagator.T%03d.D%01d.V%03d.%04d", time,
                dirac, vec, conf);
          } else {
            sprintf(propagatorfile,
                "propagator.%s.R%03d.T%03d.D%01d.V%03d.%04d",
                (dilution_list[dilution].quark == D_UP) ? "u" : "d", dilution,
                time, dirac, vec, conf);
          }
#if DEBUG
#ifdef OMP
          printf("thread %d writing file %s\n", omp_get_thread_num(),
              propagatorfile);
#else
          printf("writing file %s\n", propagatorfile);
#endif
#endif
          if ((file = fopen(propagatorfile, "wb")) == NULL ) {
            fprintf(stderr, "could not open propagator file %s.\nAborting...\n",
                propagatorfile);
            exit(-1);
          }
          count = fwrite(inverted, sizeof(spinor), VOLUMEPLUSRAND, file);
          if (count != VOLUMEPLUSRAND) {
            fprintf(stderr, "could not write all data to file %s.\n",
                propagatorfile);
          }
          fflush(file);
          fclose(file);

        } // dirac
      } // LapH
    } // time

    free(even);
    free(odd);
    free(inverted);

#ifdef OMP
  }
#endif
  return;
}

/*
 * create the perambulators
 */
void create_perambulators(int const conf, int const dilution) {
  // set the correct parameters for the loops
  int t_end = -1, l_end = -1, d_end = 4, pwidth = 0;
  if (g_stochastical_run == 0) {
    t_end = T;
    d_end = 4;
    l_end = no_eigenvalues;
  } else { // dilution in spin space not implemented!
    if (dilution_list[dilution].type[0] == D_FULL) {
      t_end = T;
    } else if (dilution_list[dilution].type[0] == D_INTER
        || dilution_list[dilution].type[0] == D_BLOCK) {
      t_end = dilution_list[dilution].size[0];
    } else if (dilution_list[dilution].type[0] == D_NONE) {
      t_end = 1;
    } else if (dilution_list[dilution].type[0] == D_INTERBLOCK) {
      t_end = T / 4;
    }

    if (dilution_list[dilution].type[1] == D_FULL) {
      d_end = 4;
    } else if (dilution_list[dilution].type[1] == D_NONE) {
      d_end = 1;
    }

    if (dilution_list[dilution].type[2] == D_FULL) {
      l_end = no_eigenvalues;
    } else if (dilution_list[dilution].type[2] == D_INTER
        || dilution_list[dilution].type[2] == D_BLOCK) {
      l_end = dilution_list[dilution].size[2];
    } else if (dilution_list[dilution].type[2] == D_NONE) {
      l_end = 1;
    }
  }
#if ONESLICE
  t_end = 1;
#endif
  pwidth = t_end * l_end * d_end;
#if DEBUG
  printf("\nparameters for perambulator: t %d, d %d, l %d\n", t_end, d_end,
      l_end);
#endif

  // read the eigenvectors
  char eigenvectorfile[200];
  su3_vector *eigenvectors = NULL;
  eigenvectors = calloc(VOLUME * no_eigenvalues, sizeof(su3_vector));
  if (eigenvectors == NULL ) {
    fprintf(stderr, "Could not allocate eigenvectors!\nAborting...\n");
    exit(-1);
  }

#if BINARYINPUT
  for (int t = 0; t < T; t++) {
    sprintf(eigenvectorfile, "%seigenvectors.%04d.%03d", EIGENSYSTEMPATH, conf,
        t);
#if DEBUG
    printf("reading file %s\n", eigenvectorfile);
#endif
    read_binary_eigenvector(&(eigenvectors[t * no_eigenvalues * SPACEVOLUME]),
        eigenvectorfile);
  }
#else
  for (int t = 0; t < T; t++) {
    for (int v = 0; v < no_eigenvalues; v++) {
      sprintf(eigenvectorfile, "%seigenvector.%03d.%03d.%04d", EIGENSYSTEMPATH,
          v, t, conf);
#if DEBUG
      printf("reading file %s\n", eigenvectorfile);
#endif
      read_su3_vector(&(eigenvectors[(t * no_eigenvalues + v) * SPACEVOLUME]),
          eigenvectorfile, 0, t, 1);
    }
  }
#endif

#ifdef OMP
#pragma omp parallel \
    shared(t_end, l_end, d_end, pwidth, eigenvectors)
  {
#endif
    int count = 0, tid = 0, x = 0, thelp;
    int tsource = 0, tsink = 0, dsource = 0, lsource = 0, lsink = 0;
    spinor *inverted = NULL, *even = NULL, *odd = NULL, *tmp = NULL;
    _Complex double *perambulator = NULL;
    char invertedfile[200], perambulatorfile[200];
    FILE *file = NULL;

#ifdef OMP
    tid = omp_get_thread_num();
#endif

    // allocate the needed memory for spinors, eigenvectors and peramulator
    perambulator = (_Complex double*) calloc(
        no_eigenvalues * T * 4 * t_end * l_end * d_end,
        sizeof(_Complex double));
    if (perambulator == NULL ) {
      free(eigenvectors);
      fprintf(stderr, "Could not allocate perambulator!\nAborting...\n");
      exit(-1);
    }

    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(perambulator);
      free(eigenvectors);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    inverted =
        (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    inverted = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(perambulator);
      free(eigenvectors);
      free(inverted);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    even = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    even = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(perambulator);
      free(eigenvectors);
      free(inverted);
      free(even);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    odd = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    odd = tmp;
#endif

    // save each perambulator into new file
    // iterate over time
#ifdef OMP
#pragma omp for private(tsource)
#endif
    for (tsource = 0; tsource < t_end; tsource++) {
      // iterate over the LapH space
      for (lsource = 0; lsource < l_end; lsource++) {
        // iterate over the dirac space
        for (dsource = 0; dsource < d_end; dsource++) {
          // read in inverted source
          sprintf(invertedfile, "source%d.%04d.%02d.%02d.inverted", dsource,
              conf, tsource, lsource);
#if DEBUG
#ifdef OMP
          printf("thread %d reading file %s\n", tid, invertedfile);
#else
          printf("reading file %s\n", invertedfile);
#endif
#endif
          read_spinor(even, odd, invertedfile, 0);
          convert_eo_to_lexic(inverted, even, odd);

          // multiply propagator with eigenvectors
          for (tsink = 0; tsink < T; tsink++) {
            for (lsink = 0; lsink < no_eigenvalues; lsink++) {
              // helper variables
              int xhelp = tsource * d_end * l_end + lsource * d_end + dsource;
              // no explicit dirac index in y diraction,
              // that is done implicit in the multiplication
              int yhelp = tsink * 4 * no_eigenvalues + lsink * 4;

              for (x = 0; x < SPACEVOLUME; x++) {
                vectorcjg_times_spinor(&(perambulator[xhelp + pwidth * yhelp]),
                    eigenvectors[(tsink * no_eigenvalues + lsink) * SPACEVOLUME
                        + x], inverted[tsink * SPACEVOLUME + x], pwidth);
              }
            }
          }

        } // dirac
      } // LapH
    } // time

    // save to file
    if (g_stochastical_run == 0) {
      sprintf(perambulatorfile,
          "perambulator.Tso%03d.Dso%01d.Vso%03d.Tsi%03d.Dsi%01d.Vsi%03d.%04d",
          t_end, d_end, l_end, T, 4, no_eigenvalues, conf);
    } else {
      sprintf(perambulatorfile,
          "perambulator.dil%02d.%s.Tso%03d.Dso%01d.Vso%03d.Tsi%03d.Dsi%01d.Vsi%03d.%04d",
          dilution, dilution_list[dilution].quarktype, t_end, d_end, l_end, T,
          4, no_eigenvalues, conf);
    }
#if DEBUG
#ifdef OMP
    printf("thread %d writing file %s\n", tid, perambulatorfile);
#else
    printf("writing file %s\n", perambulatorfile);
#endif
#endif
    if ((file = fopen(perambulatorfile, "wb")) == NULL ) {
      fprintf(stderr, "could not open propagator file %s.\nAborting...\n",
          perambulatorfile);
      exit(-1);
    }
    count = fwrite(perambulator, sizeof(_Complex double),
        pwidth * T * 4 * no_eigenvalues, file);
    if (count != (pwidth * T * 4 * no_eigenvalues)) {
      fprintf(stderr, "could not write all data to file %s.\n",
          perambulatorfile);
    }
    fflush(file);
    fclose(file);

    free(even);
    free(odd);
    free(inverted);
    free(eigenvectors);
    free(perambulator);

#ifdef OMP
  }
#endif
  return;
}

/*
 * create the perambulators with stochastic sink
 */
void create_stochastic_perambulators(int const conf, int const dilution) {
// set the correct parameters for the loops
  int t_end = -1, l_end = -1, d_end = 4, pwidth = 0;
  if (g_stochastical_run == 0) {
    t_end = T;
    d_end = 4;
    l_end = no_eigenvalues;
  } else { // dilution in spin space not implemented!
    if (dilution_list[dilution].type[0] == D_FULL) {
      t_end = T;
    } else if (dilution_list[dilution].type[0] == D_INTER
        || dilution_list[dilution].type[0] == D_BLOCK) {
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
    } else if (dilution_list[dilution].type[2] == D_INTER
        || dilution_list[dilution].type[2] == D_BLOCK) {
      l_end = dilution_list[dilution].size[2];
    } else if (dilution_list[dilution].type[2] == D_NONE) {
      l_end = 1;
    }
  }
  pwidth = t_end * l_end * d_end;
#if DEBUG
  printf("\nparameters for perambulator: t %d, d %d, l %d\n", t_end, d_end,
      l_end);
#endif

  spinor *randomvectors = NULL;
  int count = 0, x = 0, thelp;
  int tsource = 0, tsink = 0, dsource = 0, lsource = 0, xsink = 0;
  spinor *inverted = NULL, *even = NULL, *odd = NULL, *tmp = NULL;
  spinor *perambulator = NULL;
  char invertedfile[200], perambulatorfile[200];
  FILE *file = NULL;
  int xblockmax = 32, xblock = 0;
  int xhelp = 0, yhelp = 0;

// allocate the needed memory for spinors
  tmp = (spinor*) calloc(l_end * t_end * d_end * xblockmax * T + 1,
      sizeof(spinor));
  if (tmp == NULL ) {
    fprintf(stderr,
        "Could not allocate spinor perambulator in create_stochastic_perambulator!\nAborting...\n");
    exit(-1);
  }
#if (defined SSE || defined SSE2 || defined SSE3)
  perambulator = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE)
      & ~ALIGN_BASE);
#else
  perambulator = tmp;
#endif
  tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
  if (tmp == NULL ) {
    free(perambulator);
    fprintf(stderr,
        "Could not allocate spinor randomvectors in create_stochastic_perambulator!\nAborting...\n");
    exit(-1);
  }
#if (defined SSE || defined SSE2 || defined SSE3)
  randomvectors = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE)
      & ~ALIGN_BASE);
#else
  randomvectors = tmp;
#endif
  tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
  if (tmp == NULL ) {
    free(perambulator);
    free(randomvectors);
    fprintf(stderr,
        "Could not allocate spinor inverted in create_stochastic_perambulator!\nAborting...\n");
    exit(-1);
  }
#if (defined SSE || defined SSE2 || defined SSE3)
  inverted = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
  inverted = tmp;
#endif
  tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
  if (tmp == NULL ) {
    free(perambulator);
    free(randomvectors);
    free(inverted);
    fprintf(stderr,
        "Could not allocate spinor even in create_stochastic_perambulator!\nAborting...\n");
    exit(-1);
  }
#if (defined SSE || defined SSE2 || defined SSE3)
  even = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
  even = tmp;
#endif
  tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
  if (tmp == NULL ) {
    free(perambulator);
    free(randomvectors);
    free(inverted);
    free(even);
    fprintf(stderr,
        "Could not allocate spinor odd in create_stochastic_perambulator!\nAborting...\n");
    exit(-1);
  }
#if (defined SSE || defined SSE2 || defined SSE3)
  odd = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
  odd = tmp;
#endif

  for (xblock = xblockmax; xblock >= 8; xblock /= 2) {
    // block structure
    memset(perambulator, 0,
        l_end * t_end * d_end * xblockmax * T * sizeof(spinor));
    rnd_z2_vector((_Complex double*) randomvectors, 12 * VOLUMEPLUSRAND); // spinor has 12 complex entries

    for (tsource = 0; tsource < t_end; tsource++) {
      // iterate over the LapH space
      for (lsource = 0; lsource < l_end; lsource++) {
        // iterate over the dirac space
        for (dsource = 0; dsource < d_end; dsource++) {
          // read in inverted source
          sprintf(invertedfile, "source%d.%04d.%02d.%02d.inverted", dsource,
              conf, tsource, lsource);
#if DEBUG
          printf("reading file %s\n", invertedfile);
#endif
          read_spinor(even, odd, invertedfile, 0);
          convert_eo_to_lexic(inverted, even, odd);
          // helper variables
          xhelp = tsource * d_end * l_end + lsource * d_end + dsource;

          // multiply propagator with randomvectors
          for (tsink = 0; tsink < T; tsink++) {
            for (xsink = 0; xsink < xblock; xsink++) {
              yhelp = tsink * xblock + xsink;

              for (x = xsink * (LX * LY * LZ / xblock);
                  x < (xsink + 1) * (LX * LY * LZ / xblock); x++) { // block structure
                spinor_times_spinor(perambulator[xhelp + pwidth * yhelp],
                    randomvectors[tsink * SPACEVOLUME + x],
                    inverted[tsink * SPACEVOLUME + x]);
              }
            }
          }

        } // dirac
      } // LapH
    } // time

    // save to file
    sprintf(perambulatorfile,
        "stoch_perambulator_block.dil%02d.%s.Tso%03d.Dso%01d.Vso%03d.Tsi%03d.Dsi%01d.Csi%01d.Xsi%03d.%04d",
        dilution, dilution_list[dilution].quarktype, t_end, d_end, l_end, T, 4,
        3, xblock, conf);
#if DEBUG
    printf("writing file %s\n", perambulatorfile);
#endif
    if ((file = fopen(perambulatorfile, "wb")) == NULL ) {
      fprintf(stderr, "could not open propagator file %s.\nAborting...\n",
          perambulatorfile);
      exit(-1);
    }
    count = fwrite(perambulator, sizeof(spinor),
        l_end * t_end * d_end * xblock * T, file);
    if (count != (l_end * t_end * d_end * xblock * T)) {
      fprintf(stderr, "could not write all data to file %s.\n",
          perambulatorfile);
    }
    fflush(file);
    fclose(file);

    sprintf(perambulatorfile,
        "stoch_randomvector_block.dil%02d.%s.Xsi%03d.%04d", dilution,
        dilution_list[dilution].quarktype, xblock, conf);
#if DEBUG
    printf("writing file %s\n", perambulatorfile);
#endif
    if ((file = fopen(perambulatorfile, "wb")) == NULL ) {
      fprintf(stderr, "could not open propagator file %s.\nAborting...\n",
          perambulatorfile);
      exit(-1);
    }
    count = fwrite(randomvectors, sizeof(spinor), VOLUMEPLUSRAND, file);
    if (count != (VOLUMEPLUSRAND)) {
      fprintf(stderr, "could not write all data to file %s.\n",
          perambulatorfile);
    }
    fflush(file);
    fclose(file);

    // interlace structure
    memset(perambulator, 0,
        l_end * t_end * d_end * xblockmax * T * sizeof(spinor));
    rnd_z2_vector((_Complex double*) randomvectors, 12 * VOLUMEPLUSRAND); // spinor has 12 complex entries

    for (tsource = 0; tsource < t_end; tsource++) {
      // iterate over the LapH space
      for (lsource = 0; lsource < l_end; lsource++) {
        // iterate over the dirac space
        for (dsource = 0; dsource < d_end; dsource++) {
          // read in inverted source
          sprintf(invertedfile, "source%d.%04d.%02d.%02d.inverted", dsource,
              conf, tsource, lsource);
#if DEBUG
          printf("reading file %s\n", invertedfile);
#endif
          read_spinor(even, odd, invertedfile, 0);
          convert_eo_to_lexic(inverted, even, odd);
          // helper variables
          xhelp = tsource * d_end * l_end + lsource * d_end + dsource;

          // multiply propagator with randomvectors
          for (tsink = 0; tsink < T; tsink++) {
            for (xsink = 0; xsink < xblock; xsink++) {
              yhelp = tsink * xblock + xsink;

              for (x = xsink; x < SPACEVOLUME; x += xblock) { // interlace structure
                spinor_times_spinor(perambulator[xhelp + pwidth * yhelp],
                    randomvectors[tsink * SPACEVOLUME + x],
                    inverted[tsink * SPACEVOLUME + x]);
              }
            }
          }

        } // dirac
      } // LapH
    } // time

    // save to file
    sprintf(perambulatorfile,
        "stoch_perambulator_inter.dil%02d.%s.Tso%03d.Dso%01d.Vso%03d.Tsi%03d.Dsi%01d.Csi%01d.Xsi%03d.%04d",
        dilution, dilution_list[dilution].quarktype, t_end, d_end, l_end, T, 4,
        3, xblock, conf);
#if DEBUG
    printf("writing file %s\n", perambulatorfile);
#endif
    if ((file = fopen(perambulatorfile, "wb")) == NULL ) {
      fprintf(stderr, "could not open propagator file %s.\nAborting...\n",
          perambulatorfile);
      exit(-1);
    }
    count = fwrite(perambulator, sizeof(spinor),
        l_end * t_end * d_end * xblock * T, file);
    if (count != (l_end * t_end * d_end * xblock * T)) {
      fprintf(stderr, "could not write all data to file %s.\n",
          perambulatorfile);
    }
    fflush(file);
    fclose(file);

    sprintf(perambulatorfile,
        "stoch_randomvector_inter.dil%02d.%s.Xsi%03d.%04d", dilution,
        dilution_list[dilution].quarktype, xblock, conf);
#if DEBUG
    printf("writing file %s\n", perambulatorfile);
#endif
    if ((file = fopen(perambulatorfile, "wb")) == NULL ) {
      fprintf(stderr, "could not open propagator file %s.\nAborting...\n",
          perambulatorfile);
      exit(-1);
    }
    count = fwrite(randomvectors, sizeof(spinor), VOLUMEPLUSRAND, file);
    if (count != (VOLUMEPLUSRAND)) {
      fprintf(stderr, "could not write all data to file %s.\n",
          perambulatorfile);
    }
    fflush(file);
    fclose(file);
  }

  free(even);
  free(odd);
  free(inverted);
  free(randomvectors);
  free(perambulator);

  return;
}
