/***********************************************************************
 *
 * Copyright (C) 2013 Christian Jost
 *
 * This file is part of tmLQCD.
 *
 * tmLQCD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * tmLQCD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with tmLQCD.  If not, see <http://www.gnu.org/licenses/>.
 ***********************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#ifdef MPI
# include <mpi.h>
#endif
#ifdef OMP
#include <omp.h>
#endif
#include "global.h"
#include "default_input_values.h"
#include "read_input.h"
#include "start.h"
#include "dilution.h"

#include "operator.h"
#include "io/utils.h"
#include "io/spinor.h"
#include "io/su3_vector.h"
#include "linalg/convert_eo_to_lexic.h"
#include "ranlxd.h"

#define DEBUG 1
int g_stochastical_run = 1;
int no_dilution = 0;
dilution dilution_list[max_no_dilution];

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

void static create_input_files(int const dirac, int const timeslice,
    int const conf, int const nr_dilution, int const thread) {
  char filename[150];
  FILE* file = NULL;
  int j = 0, n_op = 0;
  for (j = 0; j < dirac; j++) {
    sprintf(filename, "dirac%d.%d-cg.input", j, thread);
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
    fprintf(file, "DisableIOChecks = yes\n");
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
      if (dilution_list[nr_dilution].type[2] == D_FULL) {
        fprintf(file, "Indices = 0-%d\n\n", no_eigenvalues - 1);
      } else if (dilution_list[nr_dilution].type[2] == D_NONE) {
        fprintf(file, "Indices = 0\n\n");
      } else if (dilution_list[nr_dilution].type[2] == D_INTER) {
        fprintf(file, "Indices = 0-%d\n\n",
            dilution_list[nr_dilution].size[2] - 1);
      } else if (dilution_list[nr_dilution].type[2] == D_BLOCK) {
        fprintf(file, "Indices = 0-%d\n\n",
            dilution_list[nr_dilution].size[2] - 1);
      }
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

// TODO change the call of the inverter in the routines,
// range of vec is covered by the inverter itself

void create_source_tf_df_lf(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  char call[150];
  int tslice = 0, vec = 0, point = 0, j = 0;
  int status = 0, block = LX * LY * LZ;
  int rnd_vec_size = T * 4 * no_eigenvalues;
  int count, tid = 0;
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *even = NULL, *odd = NULL;
  spinor *dirac0 = NULL;
  spinor *dirac1 = NULL;
  spinor *dirac2 = NULL;
  spinor *dirac3 = NULL;
  WRITER* writer = NULL;
  _Complex double *rnd_vector = NULL;
  FILE* file;
  int index = 0;

  rnd_vector = (_Complex double*) calloc(rnd_vec_size, sizeof(_Complex double));
  if (rnd_vector == NULL ) {
    fprintf(stderr, "Could not allocate random vector!\nAborting...\n");
    exit(-1);
  }
  rnd_z2_vector(rnd_vector, rnd_vec_size);
  sprintf(filename, "randomvector.%03d.%04d", nr_dilution, nr_conf);
#if DEBUG
  printf("\nwriting random vector to file %s\n", filename);
#endif
  if ((file = fopen(filename, "wb")) == NULL ) {
    fprintf(stderr, "Could not open file %s for random vector\nAborting...\n",
        filename);
    exit(-1);
  }
  count = fwrite(rnd_vector, sizeof(_Complex double), rnd_vec_size, file);

  if (count != rnd_vec_size) {
    fprintf(stderr,
        "Could not print all data to file %s for random vector (%d of %d)\nAborting...\n",
        filename, count, rnd_vec_size);
    exit(-1);
  }
  fclose(file);
#ifdef OMP
#pragma omp parallel \
		private(tmp, even, odd, dirac0, dirac1, dirac2, dirac3, eigenvector, \
		  writer, file, vec, j, tid, point, index, filename, status, call) \
		shared(rnd_vector, block, inverterpath)
  {
    tid = omp_get_thread_num();
#endif
    //  allocate spinors and eigenvectors
    eigenvector = (su3_vector*) calloc(block, sizeof(su3_vector));
    if (eigenvector == NULL ) {
      free(rnd_vector);
      fprintf(stderr, "Could not allocate eigenvector!\nAborting...\n");
      exit(-1);
    }

    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac0 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac0 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac1 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac1 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac2 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac2 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac3 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac3 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      free(dirac3);
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
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      free(dirac3);
      free(even);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    odd = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    odd = tmp;
#endif

#ifdef OMP
#pragma omp for private(tslice)
#endif
    for (tslice = 0; tslice < T; tslice++) {
      for (vec = 0; vec < no_eigenvalues; vec++) {
        // zero the spinor fields
        zero_spinor_field(dirac0, VOLUMEPLUSRAND);
        zero_spinor_field(dirac1, VOLUMEPLUSRAND);
        zero_spinor_field(dirac2, VOLUMEPLUSRAND);
        zero_spinor_field(dirac3, VOLUMEPLUSRAND);

        // read in eigenvector and distribute it to the sources
        sprintf(filename, "./eigenvector.%03d.%03d.%04d", vec, tslice, nr_conf);
#if DEBUG
#ifdef OMP
        printf("thread %d reading file %s\n", tid, filename);
        fflush(stdout);
#else
        printf("reading file %s\n", filename);
#endif
#endif
        read_su3_vector(eigenvector, filename, 0, tslice, 1);
        index = tslice * no_eigenvalues * 4 + vec * 4;
        for (point = 0; point < block; point++) {
          _vector_add_mul( dirac0[block*tslice + point].s0, rnd_vector[index+0],
              eigenvector[point]);
          _vector_add_mul( dirac1[block*tslice + point].s1, rnd_vector[index+1],
              eigenvector[point]);
          _vector_add_mul( dirac2[block*tslice + point].s2, rnd_vector[index+2],
              eigenvector[point]);
          _vector_add_mul( dirac3[block*tslice + point].s3, rnd_vector[index+3],
              eigenvector[point]);
        }

        // write spinor field with entries at dirac 0
        convert_lexic_to_eo(even, odd, dirac0);
        sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 1
        convert_lexic_to_eo(even, odd, dirac1);
        sprintf(filename, "%s.%04d.%02d.%02d", "source1", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 2
        convert_lexic_to_eo(even, odd, dirac2);
        sprintf(filename, "%s.%04d.%02d.%02d", "source2", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 3
        convert_lexic_to_eo(even, odd, dirac3);
        sprintf(filename, "%s.%04d.%02d.%02d", "source3", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
      }
      create_input_files(4, tslice, nr_conf, nr_dilution, tid);
      for (j = 0; j < 4; j++) {
        sprintf(call, "%s -f dirac%d.%d-cg.input 1> /dev/null", inverterpath, j,
            tid);
#if DEBUG
#ifdef OMP
        printf("\n\nthread %d trying: %s for conf %d, t %d (full)\n", tid, call,
            nr_conf, tslice);
        fflush(stdout);
#else
        printf("\n\ntrying: %s for conf %d, t %d (full)\n", call,
            nr_conf, tslice);
#endif
#endif

        fflush(stdout);
        system(call);
      }
    }
    free(eigenvector);
    free(dirac0);
    free(dirac1);
    free(dirac2);
    free(dirac3);
    free(even);
    free(odd);
#ifdef OMP
  } // end of OMP region
#endif
  free(rnd_vector);
  return;
}

void create_source_tf_df_ln(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  char call[150];
  int tslice = 0, vec = 0, point = 0, j = 0;
  int status = 0, block = LX * LY * LZ;
  int rnd_vec_size = T * 4 * no_eigenvalues;
  int count, tid = 0;
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *even = NULL, *odd = NULL;
  spinor *dirac0 = NULL;
  spinor *dirac1 = NULL;
  spinor *dirac2 = NULL;
  spinor *dirac3 = NULL;
  WRITER* writer = NULL;
  _Complex double *rnd_vector = NULL;
  FILE* file;
  int index = 0;

  rnd_vector = (_Complex double*) calloc(rnd_vec_size, sizeof(_Complex double));
  if (rnd_vector == NULL ) {
    fprintf(stderr, "Could not allocate random vector!\nAborting...\n");
    exit(-1);
  }
  rnd_z2_vector(rnd_vector, rnd_vec_size);
  sprintf(filename, "randomvector.%03d.%04d", nr_dilution, nr_conf);
#if DEBUG
  printf("\nwriting random vector to file %s\n", filename);
#endif
  if ((file = fopen(filename, "wb")) == NULL ) {
    fprintf(stderr, "Could not open file %s for random vector\nAborting...\n",
        filename);
    exit(-1);
  }
  count = fwrite(rnd_vector, sizeof(_Complex double), rnd_vec_size, file);

  if (count != rnd_vec_size) {
    fprintf(stderr,
        "Could not print all data to file %s for random vector (%d of %d)\nAborting...\n",
        filename, count, rnd_vec_size);
    exit(-1);
  }
  fclose(file);
#ifdef OMP
#pragma omp parallel \
		private(tmp, even, odd, dirac0, dirac1, dirac2, dirac3, eigenvector, \
	    writer, file, vec, j, tid, point, index, filename, status, call) \
	 shared(rnd_vector, block, inverterpath)
  {
    tid = omp_get_thread_num();
#endif
    //  allocate spinors and eigenvectors
    eigenvector = (su3_vector*) calloc(block, sizeof(su3_vector));
    if (eigenvector == NULL ) {
      free(rnd_vector);
      fprintf(stderr, "Could not allocate eigenvector!\nAborting...\n");
      exit(-1);
    }

    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac0 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac0 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac1 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac1 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac2 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac2 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac3 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac3 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      free(dirac3);
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
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      free(dirac3);
      free(even);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    odd = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    odd = tmp;
#endif

#ifdef OMP
#pragma omp for private(tslice)
#endif
    for (tslice = 0; tslice < T; tslice++) {
      zero_spinor_field(dirac0, VOLUMEPLUSRAND);
      zero_spinor_field(dirac1, VOLUMEPLUSRAND);
      zero_spinor_field(dirac2, VOLUMEPLUSRAND);
      zero_spinor_field(dirac3, VOLUMEPLUSRAND);

      for (vec = 0; vec < no_eigenvalues; vec++) {
        // read in eigenvector and distribute it to the sources
        sprintf(filename, "./eigenvector.%03d.%03d.%04d", vec, tslice, nr_conf);
#if DEBUG
#ifdef OMP
        printf("thread %d reading file %s\n", tid, filename);
        fflush(stdout);
#else
        printf("reading file %s\n", filename);
#endif
#endif
        read_su3_vector(eigenvector, filename, 0, tslice, 1);
        index = tslice * no_eigenvalues * 4 + vec * 4;
        for (point = 0; point < block; point++) {
          _vector_add_mul( dirac0[block*tslice + point].s0, rnd_vector[index+0],
              eigenvector[point]);
          _vector_add_mul( dirac1[block*tslice + point].s1, rnd_vector[index+1],
              eigenvector[point]);
          _vector_add_mul( dirac2[block*tslice + point].s2, rnd_vector[index+2],
              eigenvector[point]);
          _vector_add_mul( dirac3[block*tslice + point].s3, rnd_vector[index+3],
              eigenvector[point]);
        }
      }

      // write spinor field with entries at dirac 0
      convert_lexic_to_eo(even, odd, dirac0);
      sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, 0);
#if DEBUG
#ifdef OMP
      printf("thread %d writing file %s\n", tid, filename);
#else
      printf("writing file %s\n", filename);
#endif
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &even, &odd, 1, 64);
      destruct_writer(writer);
      // write spinor field with entries at dirac 1
      convert_lexic_to_eo(even, odd, dirac1);
      sprintf(filename, "%s.%04d.%02d.%02d", "source1", nr_conf, tslice, 0);
#if DEBUG
#ifdef OMP
      printf("thread %d writing file %s\n", tid, filename);
#else
      printf("writing file %s\n", filename);
#endif
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &even, &odd, 1, 64);
      destruct_writer(writer);
      // write spinor field with entries at dirac 2
      convert_lexic_to_eo(even, odd, dirac2);
      sprintf(filename, "%s.%04d.%02d.%02d", "source2", nr_conf, tslice, 0);
#if DEBUG
#ifdef OMP
      printf("thread %d writing file %s\n", tid, filename);
#else
      printf("writing file %s\n", filename);
#endif
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &even, &odd, 1, 64);
      destruct_writer(writer);
      // write spinor field with entries at dirac 3
      convert_lexic_to_eo(even, odd, dirac3);
      sprintf(filename, "%s.%04d.%02d.%02d", "source3", nr_conf, tslice, 0);
#if DEBUG
#ifdef OMP
      printf("thread %d writing file %s\n", tid, filename);
#else
      printf("writing file %s\n", filename);
#endif
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &even, &odd, 1, 64);
      destruct_writer(writer);

      create_input_files(4, tslice, nr_conf, nr_dilution, tid);
      for (j = 0; j < 4; j++) {
        sprintf(call, "%s -f dirac%d.%d-cg.input 1> /dev/null", inverterpath, j,
            tid);
#if DEBUG
#ifdef OMP
        printf("\n\nthread %d trying: %s for conf %d, t %d (full)\n", tid, call,
            nr_conf, tslice);
        fflush(stdout);
#else
        printf("\n\ntrying: %s for conf %d, t %d (full)\n", call,
            nr_conf, tslice);
#endif
#endif
        fflush(stdout);
        system(call);
      }
    }
    free(eigenvector);
    free(dirac0);
    free(dirac1);
    free(dirac2);
    free(dirac3);
    free(even);
    free(odd);
#ifdef OMP
  } // end of OMP region
#endif
  free(rnd_vector);
  return;
}

void create_source_tf_df_li(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  char call[150];
  int tslice = 0, vec = 0, point = 0, j = 0;
  int status = 0, block = LX * LY * LZ;
  int rnd_vec_size = T * 4 * no_eigenvalues;
  int count, tid = 0, v = 0;
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *even = NULL, *odd = NULL;
  spinor *dirac0 = NULL;
  spinor *dirac1 = NULL;
  spinor *dirac2 = NULL;
  spinor *dirac3 = NULL;
  WRITER* writer = NULL;
  _Complex double *rnd_vector = NULL;
  FILE* file;
  int index = 0;

  rnd_vector = (_Complex double*) calloc(rnd_vec_size, sizeof(_Complex double));
  if (rnd_vector == NULL ) {
    fprintf(stderr, "Could not allocate random vector!\nAborting...\n");
    exit(-1);
  }
  rnd_z2_vector(rnd_vector, rnd_vec_size);
  sprintf(filename, "randomvector.%03d.%04d", nr_dilution, nr_conf);
#if DEBUG
  printf("\nwriting random vector to file %s\n", filename);
#endif
  if ((file = fopen(filename, "wb")) == NULL ) {
    fprintf(stderr, "Could not open file %s for random vector\nAborting...\n",
        filename);
    exit(-1);
  }
  count = fwrite(rnd_vector, sizeof(_Complex double), rnd_vec_size, file);

  if (count != rnd_vec_size) {
    fprintf(stderr,
        "Could not print all data to file %s for random vector (%d of %d)\nAborting...\n",
        filename, count, rnd_vec_size);
    exit(-1);
  }
  fclose(file);
#ifdef OMP
#pragma omp parallel \
		private(tmp, even, odd, dirac0, dirac1, dirac2, dirac3, eigenvector, \
		  writer, file, vec, j, tid, v, point, index, filename, status, call) \
		shared(rnd_vector, block, inverterpath)
  {
    tid = omp_get_thread_num();
#endif
    //  allocate spinors and eigenvectors
    eigenvector = (su3_vector*) calloc(block, sizeof(su3_vector));
    if (eigenvector == NULL ) {
      free(rnd_vector);
      fprintf(stderr, "Could not allocate eigenvector!\nAborting...\n");
      exit(-1);
    }

    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac0 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac0 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac1 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac1 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac2 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac2 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac3 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac3 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      free(dirac3);
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
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      free(dirac3);
      free(even);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    odd = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    odd = tmp;
#endif

#ifdef OMP
#pragma omp for private(tslice)
#endif
    for (tslice = 0; tslice < T; tslice++) {
      for (vec = 0; vec < dilution_list[nr_dilution].size[2]; vec++) {
        // zero the spinor fields
        zero_spinor_field(dirac0, VOLUMEPLUSRAND);
        zero_spinor_field(dirac1, VOLUMEPLUSRAND);
        zero_spinor_field(dirac2, VOLUMEPLUSRAND);
        zero_spinor_field(dirac3, VOLUMEPLUSRAND);

        for (v = vec; v < vec + dilution_list[nr_dilution].size[2]; v++) {
          // read in eigenvector and distribute it to the sources
          sprintf(filename, "./eigenvector.%03d.%03d.%04d", v, tslice, nr_conf);
#if DEBUG
#ifdef OMP
          printf("thread %d reading file %s\n", tid, filename);
          fflush(stdout);
#else
          printf("reading file %s\n", filename);
#endif
#endif
          read_su3_vector(eigenvector, filename, 0, tslice, 1);
          index = tslice * no_eigenvalues * 4 + v * 4;
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

        // write spinor field with entries at dirac 0
        convert_lexic_to_eo(even, odd, dirac0);
        sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 1
        convert_lexic_to_eo(even, odd, dirac1);
        sprintf(filename, "%s.%04d.%02d.%02d", "source1", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 2
        convert_lexic_to_eo(even, odd, dirac2);
        sprintf(filename, "%s.%04d.%02d.%02d", "source2", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 3
        convert_lexic_to_eo(even, odd, dirac3);
        sprintf(filename, "%s.%04d.%02d.%02d", "source3", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
      }

      create_input_files(4, tslice, nr_conf, nr_dilution, tid);
      for (j = 0; j < 4; j++) {
        sprintf(call, "%s -f dirac%d.%d-cg.input 1> /dev/null", inverterpath, j,
            tid);
#if DEBUG
#ifdef OMP
        printf("\n\nthread %d trying: %s for conf %d, t %d (full)\n", tid, call,
            nr_conf, tslice);
        fflush(stdout);
#else
        printf("\n\ntrying: %s for conf %d, t %d (full)\n", call,
            nr_conf, tslice);
#endif
#endif
        fflush(stdout);
        system(call);
      }
    }
    free(eigenvector);
    free(dirac0);
    free(dirac1);
    free(dirac2);
    free(dirac3);
    free(even);
    free(odd);
#ifdef OMP
  } // end of OMP region
#endif
  free(rnd_vector);
  return;
}

void create_source_tf_df_li1(const int nr_conf, const int nr_dilution,
    char* inverterpath, int *tslices, int nr_tslices) {
  char filename[200];
  char call[150];
  int tslice = 0, vec = 0, point = 0, j = 0, i = 0;
  int status = 0, block = LX * LY * LZ;
  int rnd_vec_size = T * 4 * no_eigenvalues;
  int count, tid = 0, v = 0;
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *even = NULL, *odd = NULL;
  spinor *dirac0 = NULL;
  spinor *dirac1 = NULL;
  spinor *dirac2 = NULL;
  spinor *dirac3 = NULL;
  WRITER* writer = NULL;
  _Complex double *rnd_vector = NULL;
  FILE* file;
  int index = 0;

  rnd_vector = (_Complex double*) calloc(rnd_vec_size, sizeof(_Complex double));
  if (rnd_vector == NULL ) {
    fprintf(stderr, "Could not allocate random vector!\nAborting...\n");
    exit(-1);
  }
  rnd_z2_vector(rnd_vector, rnd_vec_size);
  sprintf(filename, "randomvector.%03d.%04d", nr_dilution, nr_conf);
#if DEBUG
  printf("\nwriting random vector to file %s\n", filename);
#endif
  if ((file = fopen(filename, "wb")) == NULL ) {
    fprintf(stderr, "Could not open file %s for random vector\nAborting...\n",
        filename);
    exit(-1);
  }
  count = fwrite(rnd_vector, sizeof(_Complex double), rnd_vec_size, file);

  if (count != rnd_vec_size) {
    fprintf(stderr,
        "Could not print all data to file %s for random vector (%d of %d)\nAborting...\n",
        filename, count, rnd_vec_size);
    exit(-1);
  }
  fclose(file);
#ifdef OMP
#pragma omp parallel \
    private(tmp, even, odd, dirac0, dirac1, dirac2, dirac3, eigenvector, \
      writer, file, vec, i, j, tid, v, point, index, filename, status, call, tslice) \
    shared(rnd_vector, block, inverterpath)
  {
    tid = omp_get_thread_num();
#endif
    //  allocate spinors and eigenvectors
    eigenvector = (su3_vector*) calloc(block, sizeof(su3_vector));
    if (eigenvector == NULL ) {
      free(rnd_vector);
      fprintf(stderr, "Could not allocate eigenvector!\nAborting...\n");
      exit(-1);
    }

    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac0 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac0 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac1 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac1 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac2 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac2 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac3 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac3 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      free(dirac3);
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
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      free(dirac3);
      free(even);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    odd = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    odd = tmp;
#endif

#ifdef OMP
#pragma omp for private(i, tslice)
#endif
    for (i = 0; i < nr_tslices; i++) {
      tslice = tslices[i];
      for (vec = 0; vec < dilution_list[nr_dilution].size[2]; vec++) {
        // zero the spinor fields
        zero_spinor_field(dirac0, VOLUMEPLUSRAND);
        zero_spinor_field(dirac1, VOLUMEPLUSRAND);
        zero_spinor_field(dirac2, VOLUMEPLUSRAND);
        zero_spinor_field(dirac3, VOLUMEPLUSRAND);

        for (v = vec; v < vec + dilution_list[nr_dilution].size[2]; v++) {
          // read in eigenvector and distribute it to the sources
          sprintf(filename, "./eigenvector.%03d.%03d.%04d", v, tslice, nr_conf);
#if DEBUG
#ifdef OMP
          printf("thread %d reading file %s\n", tid, filename);
          fflush(stdout);
#else
          printf("reading file %s\n", filename);
#endif
#endif
          read_su3_vector(eigenvector, filename, 0, tslice, 1);
          index = tslice * no_eigenvalues * 4 + v * 4;
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

        // write spinor field with entries at dirac 0
        convert_lexic_to_eo(even, odd, dirac0);
        sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 1
        convert_lexic_to_eo(even, odd, dirac1);
        sprintf(filename, "%s.%04d.%02d.%02d", "source1", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 2
        convert_lexic_to_eo(even, odd, dirac2);
        sprintf(filename, "%s.%04d.%02d.%02d", "source2", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 3
        convert_lexic_to_eo(even, odd, dirac3);
        sprintf(filename, "%s.%04d.%02d.%02d", "source3", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
      }

      create_input_files(4, tslice, nr_conf, nr_dilution, tid);
      for (j = 0; j < 4; j++) {
        sprintf(call, "%s -f dirac%d.%d-cg.input 1> /dev/null", inverterpath, j,
            tid);
#if DEBUG
#ifdef OMP
        printf("\n\nthread %d trying: %s for conf %d, t %d (full)\n", tid, call,
            nr_conf, tslice);
        fflush(stdout);
#else
        printf("\n\ntrying: %s for conf %d, t %d (full)\n", call,
            nr_conf, tslice);
#endif
#endif
        fflush(stdout);
        system(call);
      }
    }
    free(eigenvector);
    free(dirac0);
    free(dirac1);
    free(dirac2);
    free(dirac3);
    free(even);
    free(odd);
#ifdef OMP
  } // end of OMP region
#endif
  free(rnd_vector);
  return;
}

void create_source_tf_df_lb(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  char call[150];
  int tslice = 0, vec = 0, point = 0, j = 0;
  int status = 0, block = LX * LY * LZ;
  int rnd_vec_size = T * 4 * no_eigenvalues;
  int count, tid = 0, v = 0, vs = dilution_list[nr_dilution].size[2];
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *even = NULL, *odd = NULL;
  spinor *dirac0 = NULL;
  spinor *dirac1 = NULL;
  spinor *dirac2 = NULL;
  spinor *dirac3 = NULL;
  WRITER* writer = NULL;
  _Complex double *rnd_vector = NULL;
  FILE* file;
  int index = 0;

  rnd_vector = (_Complex double*) calloc(rnd_vec_size, sizeof(_Complex double));
  if (rnd_vector == NULL ) {
    fprintf(stderr, "Could not allocate random vector!\nAborting...\n");
    exit(-1);
  }
  rnd_z2_vector(rnd_vector, rnd_vec_size);
  sprintf(filename, "randomvector.%03d.%04d", nr_dilution, nr_conf);
#if DEBUG
  printf("\nwriting random vector to file %s\n", filename);
#endif
  if ((file = fopen(filename, "wb")) == NULL ) {
    fprintf(stderr, "Could not open file %s for random vector\nAborting...\n",
        filename);
    exit(-1);
  }
  count = fwrite(rnd_vector, sizeof(_Complex double), rnd_vec_size, file);

  if (count != rnd_vec_size) {
    fprintf(stderr,
        "Could not print all data to file %s for random vector (%d of %d)\nAborting...\n",
        filename, count, rnd_vec_size);
    exit(-1);
  }
  fclose(file);
#ifdef OMP
#pragma omp parallel \
		private(tmp, even, odd, dirac0, dirac1, dirac2, dirac3, eigenvector, \
	   writer, file, vec, j, tid, v, point, index, filename, status, call) \
	 shared(rnd_vector, block, inverterpath, vs)
  {
    tid = omp_get_thread_num();
#endif
    //  allocate spinors and eigenvectors
    eigenvector = (su3_vector*) calloc(block, sizeof(su3_vector));
    if (eigenvector == NULL ) {
      free(rnd_vector);
      fprintf(stderr, "Could not allocate eigenvector!\nAborting...\n");
      exit(-1);
    }

    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac0 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac0 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac1 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac1 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac2 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac2 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac3 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac3 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      free(dirac3);
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
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      free(dirac3);
      free(even);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    odd = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    odd = tmp;
#endif

#ifdef OMP
#pragma omp for private(tslice)
#endif
    for (tslice = 0; tslice < T; tslice++) {
      for (vec = 0; vec < dilution_list[nr_dilution].size[2]; vec++) {
        // zero the spinor fields
        zero_spinor_field(dirac0, VOLUMEPLUSRAND);
        zero_spinor_field(dirac1, VOLUMEPLUSRAND);
        zero_spinor_field(dirac2, VOLUMEPLUSRAND);
        zero_spinor_field(dirac3, VOLUMEPLUSRAND);

        for (v = vec * vs; v < (vec + 1) * vs; v++) {
          // read in eigenvector and distribute it to the sources
          sprintf(filename, "./eigenvector.%03d.%03d.%04d", v, tslice, nr_conf);
#if DEBUG
#ifdef OMP
          printf("thread %d reading file %s\n", tid, filename);
          fflush(stdout);
#else
          printf("reading file %s\n", filename);
#endif
#endif
          read_su3_vector(eigenvector, filename, 0, tslice, 1);
          index = tslice * no_eigenvalues * 4 + v * 4;
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

        // write spinor field with entries at dirac 0
        convert_lexic_to_eo(even, odd, dirac0);
        sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 1
        convert_lexic_to_eo(even, odd, dirac1);
        sprintf(filename, "%s.%04d.%02d.%02d", "source1", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 2
        convert_lexic_to_eo(even, odd, dirac2);
        sprintf(filename, "%s.%04d.%02d.%02d", "source2", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 3
        convert_lexic_to_eo(even, odd, dirac3);
        sprintf(filename, "%s.%04d.%02d.%02d", "source3", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
      }

      create_input_files(4, tslice, nr_conf, nr_dilution, tid);
      for (j = 0; j < 4; j++) {
        sprintf(call, "%s -f dirac%d.%d-cg.input 1> /dev/null", inverterpath, j,
            tid);
#if DEBUG
#ifdef OMP
        printf("\n\nthread %d trying: %s for conf %d, t %d (full)\n", tid, call,
            nr_conf, tslice);
        fflush(stdout);
#else
        printf("\n\ntrying: %s for conf %d, t %d (full)\n", call,
            nr_conf, tslice);
#endif
#endif
        fflush(stdout);
        system(call);
      }
    }
    free(eigenvector);
    free(dirac0);
    free(dirac1);
    free(dirac2);
    free(dirac3);
    free(even);
    free(odd);
#ifdef OMP
  } // end of OMP region
#endif
  free(rnd_vector);
  return;
}

void create_source_ti_df_lf(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  char call[150];
  int tslice = 0, vec = 0, point = 0, j = 0;
  int status = 0, block = LX * LY * LZ;
  int rnd_vec_size = T * 4 * no_eigenvalues;
  int count, tid = 0, v = 0, t = 0;
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *even = NULL, *odd = NULL;
  spinor *dirac0 = NULL;
  spinor *dirac1 = NULL;
  spinor *dirac2 = NULL;
  spinor *dirac3 = NULL;
  WRITER* writer = NULL;
  _Complex double *rnd_vector = NULL;
  FILE* file;
  int index = 0;

  rnd_vector = (_Complex double*) calloc(rnd_vec_size, sizeof(_Complex double));
  if (rnd_vector == NULL ) {
    fprintf(stderr, "Could not allocate random vector!\nAborting...\n");
    exit(-1);
  }
  rnd_z2_vector(rnd_vector, rnd_vec_size);
  sprintf(filename, "randomvector.%03d.%04d", nr_dilution, nr_conf);
#if DEBUG
  printf("\nwriting random vector to file %s\n", filename);
#endif
  if ((file = fopen(filename, "wb")) == NULL ) {
    fprintf(stderr, "Could not open file %s for random vector\nAborting...\n",
        filename);
    exit(-1);
  }
  count = fwrite(rnd_vector, sizeof(_Complex double), rnd_vec_size, file);

  if (count != rnd_vec_size) {
    fprintf(stderr,
        "Could not print all data to file %s for random vector (%d of %d)\nAborting...\n",
        filename, count, rnd_vec_size);
    exit(-1);
  }
  fclose(file);
#ifdef OMP
#pragma omp parallel \
  private(tmp, even, odd, dirac0, dirac1, dirac2, dirac3, eigenvector, \
    writer, file, vec, j, tid, v, t, point, index, filename, status, call) \
  shared(rnd_vector, block, inverterpath)
  {
    tid = omp_get_thread_num();
#endif
    //  allocate spinors and eigenvectors
    eigenvector = (su3_vector*) calloc(block, sizeof(su3_vector));
    if (eigenvector == NULL ) {
      free(rnd_vector);
      fprintf(stderr, "Could not allocate eigenvector!\nAborting...\n");
      exit(-1);
    }

    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac0 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac0 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac1 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac1 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac2 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac2 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac3 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac3 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      free(dirac3);
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
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      free(dirac3);
      free(even);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    odd = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    odd = tmp;
#endif

#ifdef OMP
#pragma omp for private(tslice)
#endif
    for (tslice = 0; tslice < dilution_list[nr_dilution].size[0]; tslice++) {
      for (vec = 0; vec < no_eigenvalues; vec++) {
        zero_spinor_field(dirac0, VOLUMEPLUSRAND);
        zero_spinor_field(dirac1, VOLUMEPLUSRAND);
        zero_spinor_field(dirac2, VOLUMEPLUSRAND);
        zero_spinor_field(dirac3, VOLUMEPLUSRAND);

        for (t = tslice; t < T; t += dilution_list[nr_dilution].size[0]) {
          // read in eigenvector and distribute it to the sources
          sprintf(filename, "./eigenvector.%03d.%03d.%04d", vec, t, nr_conf);
#if DEBUG
#ifdef OMP
          printf("thread %d reading file %s\n", tid, filename);
          fflush(stdout);
#else
          printf("reading file %s\n", filename);
#endif
#endif
          read_su3_vector(eigenvector, filename, 0, t, 1);
          index = t * no_eigenvalues * 4 + vec * 4;
          for (point = 0; point < block; point++) {
            _vector_add_mul( dirac0[block*t + point].s0, rnd_vector[index+0],
                eigenvector[point]);
            _vector_add_mul( dirac1[block*t + point].s1, rnd_vector[index+1],
                eigenvector[point]);
            _vector_add_mul( dirac2[block*t + point].s2, rnd_vector[index+2],
                eigenvector[point]);
            _vector_add_mul( dirac3[block*t + point].s3, rnd_vector[index+3],
                eigenvector[point]);
          }
        }

        // write spinor field with entries at dirac 0
        convert_lexic_to_eo(even, odd, dirac0);
        sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 1
        convert_lexic_to_eo(even, odd, dirac1);
        sprintf(filename, "%s.%04d.%02d.%02d", "source1", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 2
        convert_lexic_to_eo(even, odd, dirac2);
        sprintf(filename, "%s.%04d.%02d.%02d", "source2", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 3
        convert_lexic_to_eo(even, odd, dirac3);
        sprintf(filename, "%s.%04d.%02d.%02d", "source3", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
      }

      create_input_files(4, tslice, nr_conf, nr_dilution, tid);
      for (j = 0; j < 4; j++) {
        sprintf(call, "%s -f dirac%d.%d-cg.input 1> /dev/null", inverterpath, j,
            tid);
#if DEBUG
#ifdef OMP
        printf("\n\nthread %d trying: %s for conf %d, t %d (full)\n", tid, call,
            nr_conf, tslice);
        fflush(stdout);
#else
        printf("\n\ntrying: %s for conf %d, t %d (full)\n", call,
            nr_conf, tslice);
#endif
#endif
        fflush(stdout);
        system(call);

      }
    }
    free(eigenvector);
    free(dirac0);
    free(dirac1);
    free(dirac2);
    free(dirac3);
    free(even);
    free(odd);
#ifdef OMP
  } // end of OMP region
#endif
  free(rnd_vector);
  return;
}

void create_source_ti_df_ln(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  char call[150];
  int tslice = 0, vec = 0, point = 0, j = 0;
  int status = 0, block = LX * LY * LZ;
  int rnd_vec_size = T * 4 * no_eigenvalues;
  int count, tid = 0, v = 0, t = 0;
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *even = NULL, *odd = NULL;
  spinor *dirac0 = NULL;
  spinor *dirac1 = NULL;
  spinor *dirac2 = NULL;
  spinor *dirac3 = NULL;
  WRITER* writer = NULL;
  _Complex double *rnd_vector = NULL;
  FILE* file;
  int index = 0;

  rnd_vector = (_Complex double*) calloc(rnd_vec_size, sizeof(_Complex double));
  if (rnd_vector == NULL ) {
    fprintf(stderr, "Could not allocate random vector!\nAborting...\n");
    exit(-1);
  }
  rnd_z2_vector(rnd_vector, rnd_vec_size);
  sprintf(filename, "randomvector.%03d.%04d", nr_dilution, nr_conf);
#if DEBUG
  printf("\nwriting random vector to file %s\n", filename);
#endif
  if ((file = fopen(filename, "wb")) == NULL ) {
    fprintf(stderr, "Could not open file %s for random vector\nAborting...\n",
        filename);
    exit(-1);
  }
  count = fwrite(rnd_vector, sizeof(_Complex double), rnd_vec_size, file);

  if (count != rnd_vec_size) {
    fprintf(stderr,
        "Could not print all data to file %s for random vector (%d of %d)\nAborting...\n",
        filename, count, rnd_vec_size);
    exit(-1);
  }
  fclose(file);
#ifdef OMP
#pragma omp parallel \
  private(tmp, even, odd, dirac0, dirac1, dirac2, dirac3, eigenvector, \
    writer, file, vec, j, tid, v, t, point, index, filename, status, call) \
  shared(rnd_vector, block, inverterpath)
  {
    tid = omp_get_thread_num();
#endif
    //  allocate spinors and eigenvectors
    eigenvector = (su3_vector*) calloc(block, sizeof(su3_vector));
    if (eigenvector == NULL ) {
      free(rnd_vector);
      fprintf(stderr, "Could not allocate eigenvector!\nAborting...\n");
      exit(-1);
    }

    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac0 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac0 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac1 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac1 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac2 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac2 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac3 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac3 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      free(dirac3);
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
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      free(dirac3);
      free(even);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    odd = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    odd = tmp;
#endif

#ifdef OMP
#pragma omp for private(tslice)
#endif
    for (tslice = 0; tslice < dilution_list[nr_dilution].size[0]; tslice++) {
      zero_spinor_field(dirac0, VOLUMEPLUSRAND);
      zero_spinor_field(dirac1, VOLUMEPLUSRAND);
      zero_spinor_field(dirac2, VOLUMEPLUSRAND);
      zero_spinor_field(dirac3, VOLUMEPLUSRAND);

      for (vec = 0; vec < dilution_list[nr_dilution].size[2]; vec++) {
        for (t = tslice; t < T; t += dilution_list[nr_dilution].size[0]) {
          // read in eigenvector and distribute it to the sources
          sprintf(filename, "./eigenvector.%03d.%03d.%04d", vec, t, nr_conf);
#if DEBUG
#ifdef OMP
          printf("thread %d reading file %s\n", tid, filename);
          fflush(stdout);
#else
          printf("reading file %s\n", filename);
#endif
#endif
          read_su3_vector(eigenvector, filename, 0, t, 1);
          index = t * no_eigenvalues * 4 + vec * 4;
          for (point = 0; point < block; point++) {
            _vector_add_mul( dirac0[block*t + point].s0, rnd_vector[index+0],
                eigenvector[point]);
            _vector_add_mul( dirac1[block*t + point].s1, rnd_vector[index+1],
                eigenvector[point]);
            _vector_add_mul( dirac2[block*t + point].s2, rnd_vector[index+2],
                eigenvector[point]);
            _vector_add_mul( dirac3[block*t + point].s3, rnd_vector[index+3],
                eigenvector[point]);
          }
        }
      }

      // write spinor field with entries at dirac 0
      convert_lexic_to_eo(even, odd, dirac0);
      sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, 0);
#if DEBUG
#ifdef OMP
      printf("thread %d writing file %s\n", tid, filename);
#else
      printf("writing file %s\n", filename);
#endif
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &even, &odd, 1, 64);
      destruct_writer(writer);
      // write spinor field with entries at dirac 1
      convert_lexic_to_eo(even, odd, dirac1);
      sprintf(filename, "%s.%04d.%02d.%02d", "source1", nr_conf, tslice, 0);
#if DEBUG
#ifdef OMP
      printf("thread %d writing file %s\n", tid, filename);
#else
      printf("writing file %s\n", filename);
#endif
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &even, &odd, 1, 64);
      destruct_writer(writer);
      // write spinor field with entries at dirac 2
      convert_lexic_to_eo(even, odd, dirac2);
      sprintf(filename, "%s.%04d.%02d.%02d", "source2", nr_conf, tslice, 0);
#if DEBUG
#ifdef OMP
      printf("thread %d writing file %s\n", tid, filename);
#else
      printf("writing file %s\n", filename);
#endif
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &even, &odd, 1, 64);
      destruct_writer(writer);
      // write spinor field with entries at dirac 3
      convert_lexic_to_eo(even, odd, dirac3);
      sprintf(filename, "%s.%04d.%02d.%02d", "source3", nr_conf, tslice, 0);
#if DEBUG
#ifdef OMP
      printf("thread %d writing file %s\n", tid, filename);
#else
      printf("writing file %s\n", filename);
#endif
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &even, &odd, 1, 64);
      destruct_writer(writer);

      create_input_files(4, tslice, nr_conf, nr_dilution, tid);
      for (j = 0; j < 4; j++) {
        sprintf(call, "%s -f dirac%d.%d-cg.input 1> /dev/null", inverterpath, j,
            tid);
#if DEBUG
#ifdef OMP
        printf("\n\nthread %d trying: %s for conf %d, t %d (full)\n", tid, call,
            nr_conf, tslice);
        fflush(stdout);
#else
        printf("\n\ntrying: %s for conf %d, t %d (full)\n", call,
            nr_conf, tslice);
#endif
#endif
        fflush(stdout);
        system(call);

      }
    }
    free(eigenvector);
    free(dirac0);
    free(dirac1);
    free(dirac2);
    free(dirac3);
    free(even);
    free(odd);
#ifdef OMP
  } // end of OMP region
#endif
  free(rnd_vector);
  return;
}

void create_source_ti_df_li(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  char call[150];
  int tslice = 0, vec = 0, point = 0, j = 0;
  int status = 0, block = LX * LY * LZ;
  int rnd_vec_size = T * 4 * no_eigenvalues;
  int count, tid = 0, v = 0, t = 0;
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *even = NULL, *odd = NULL;
  spinor *dirac0 = NULL;
  spinor *dirac1 = NULL;
  spinor *dirac2 = NULL;
  spinor *dirac3 = NULL;
  WRITER* writer = NULL;
  _Complex double *rnd_vector = NULL;
  FILE* file;
  int index = 0;

  rnd_vector = (_Complex double*) calloc(rnd_vec_size, sizeof(_Complex double));
  if (rnd_vector == NULL ) {
    fprintf(stderr, "Could not allocate random vector!\nAborting...\n");
    exit(-1);
  }
  rnd_z2_vector(rnd_vector, rnd_vec_size);
  sprintf(filename, "randomvector.%03d.%04d", nr_dilution, nr_conf);
#if DEBUG
  printf("\nwriting random vector to file %s\n", filename);
#endif
  if ((file = fopen(filename, "wb")) == NULL ) {
    fprintf(stderr, "Could not open file %s for random vector\nAborting...\n",
        filename);
    exit(-1);
  }
  count = fwrite(rnd_vector, sizeof(_Complex double), rnd_vec_size, file);

  if (count != rnd_vec_size) {
    fprintf(stderr,
        "Could not print all data to file %s for random vector (%d of %d)\nAborting...\n",
        filename, count, rnd_vec_size);
    exit(-1);
  }
  fclose(file);
#ifdef OMP
#pragma omp parallel \
  private(tmp, even, odd, dirac0, dirac1, dirac2, dirac3, eigenvector, \
    writer, file, vec, j, tid, v, t, point, index, filename, status, call) \
  shared(rnd_vector, block, inverterpath)
  {
    tid = omp_get_thread_num();
#endif
    //  allocate spinors and eigenvectors
    eigenvector = (su3_vector*) calloc(block, sizeof(su3_vector));
    if (eigenvector == NULL ) {
      free(rnd_vector);
      fprintf(stderr, "Could not allocate eigenvector!\nAborting...\n");
      exit(-1);
    }

    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac0 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac0 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac1 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac1 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac2 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac2 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac3 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac3 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      free(dirac3);
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
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      free(dirac3);
      free(even);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    odd = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    odd = tmp;
#endif

#ifdef OMP
#pragma omp for private(tslice)
#endif
    for (tslice = 0; tslice < dilution_list[nr_dilution].size[0]; tslice++) {
      for (vec = 0; vec < dilution_list[nr_dilution].size[2]; vec++) {
        zero_spinor_field(dirac0, VOLUMEPLUSRAND);
        zero_spinor_field(dirac1, VOLUMEPLUSRAND);
        zero_spinor_field(dirac2, VOLUMEPLUSRAND);
        zero_spinor_field(dirac3, VOLUMEPLUSRAND);

        for (v = vec; v < no_eigenvalues; v +=
            dilution_list[nr_dilution].size[2]) {
          for (t = tslice; t < T; t += dilution_list[nr_dilution].size[0]) {
            // read in eigenvector and distribute it to the sources
            sprintf(filename, "./eigenvector.%03d.%03d.%04d", v, t, nr_conf);
#if DEBUG
#ifdef OMP
            printf("thread %d reading file %s\n", tid, filename);
            fflush(stdout);
#else
            printf("reading file %s\n", filename);
#endif
#endif
            read_su3_vector(eigenvector, filename, 0, t, 1);
            index = t * no_eigenvalues * 4 + v * 4;
            for (point = 0; point < block; point++) {
              _vector_add_mul( dirac0[block*t + point].s0, rnd_vector[index+0],
                  eigenvector[point]);
              _vector_add_mul( dirac1[block*t + point].s1, rnd_vector[index+1],
                  eigenvector[point]);
              _vector_add_mul( dirac2[block*t + point].s2, rnd_vector[index+2],
                  eigenvector[point]);
              _vector_add_mul( dirac3[block*t + point].s3, rnd_vector[index+3],
                  eigenvector[point]);
            }
          }
        }

        // write spinor field with entries at dirac 0
        convert_lexic_to_eo(even, odd, dirac0);
        sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 1
        convert_lexic_to_eo(even, odd, dirac1);
        sprintf(filename, "%s.%04d.%02d.%02d", "source1", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 2
        convert_lexic_to_eo(even, odd, dirac2);
        sprintf(filename, "%s.%04d.%02d.%02d", "source2", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 3
        convert_lexic_to_eo(even, odd, dirac3);
        sprintf(filename, "%s.%04d.%02d.%02d", "source3", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
      }

      create_input_files(4, tslice, nr_conf, nr_dilution, tid);
      for (j = 0; j < 4; j++) {
        sprintf(call, "%s -f dirac%d.%d-cg.input 1> /dev/null", inverterpath, j,
            tid);
#if DEBUG
#ifdef OMP
        printf("\n\nthread %d trying: %s for conf %d, t %d (full)\n", tid, call,
            nr_conf, tslice);
        fflush(stdout);
#else
        printf("\n\ntrying: %s for conf %d, t %d (full)\n", call,
            nr_conf, tslice);
#endif
#endif
        fflush(stdout);
        system(call);

      }
    }
    free(eigenvector);
    free(dirac0);
    free(dirac1);
    free(dirac2);
    free(dirac3);
    free(even);
    free(odd);
#ifdef OMP
  } // end of OMP region
#endif
  free(rnd_vector);
  return;
}

void create_source_ti_df_lb(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  char call[150];
  int tslice = 0, vec = 0, point = 0, j = 0;
  int status = 0, block = LX * LY * LZ;
  int rnd_vec_size = T * 4 * no_eigenvalues;
  int count, tid = 0, v = 0, t = 0;
  int vs = dilution_list[nr_dilution].size[2];
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *even = NULL, *odd = NULL;
  spinor *dirac0 = NULL;
  spinor *dirac1 = NULL;
  spinor *dirac2 = NULL;
  spinor *dirac3 = NULL;
  WRITER* writer = NULL;
  _Complex double *rnd_vector = NULL;
  FILE* file;
  int index = 0;

  rnd_vector = (_Complex double*) calloc(rnd_vec_size, sizeof(_Complex double));
  if (rnd_vector == NULL ) {
    fprintf(stderr, "Could not allocate random vector!\nAborting...\n");
    exit(-1);
  }
  rnd_z2_vector(rnd_vector, rnd_vec_size);
  sprintf(filename, "randomvector.%03d.%04d", nr_dilution, nr_conf);
#if DEBUG
  printf("\nwriting random vector to file %s\n", filename);
#endif
  if ((file = fopen(filename, "wb")) == NULL ) {
    fprintf(stderr, "Could not open file %s for random vector\nAborting...\n",
        filename);
    exit(-1);
  }
  count = fwrite(rnd_vector, sizeof(_Complex double), rnd_vec_size, file);

  if (count != rnd_vec_size) {
    fprintf(stderr,
        "Could not print all data to file %s for random vector (%d of %d)\nAborting...\n",
        filename, count, rnd_vec_size);
    exit(-1);
  }
  fclose(file);
#ifdef OMP
#pragma omp parallel \
  private(tmp, even, odd, dirac0, dirac1, dirac2, dirac3, eigenvector, \
    writer, file, vec, j, tid, v, t, point, index, filename, status, call) \
  shared(rnd_vector, block, inverterpath, vs)
  {
    tid = omp_get_thread_num();
#endif
    //  allocate spinors and eigenvectors
    eigenvector = (su3_vector*) calloc(block, sizeof(su3_vector));
    if (eigenvector == NULL ) {
      free(rnd_vector);
      fprintf(stderr, "Could not allocate eigenvector!\nAborting...\n");
      exit(-1);
    }

    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac0 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac0 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac1 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac1 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac2 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac2 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    dirac3 = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    dirac3 = tmp;
#endif
    tmp = (spinor*) calloc(VOLUMEPLUSRAND + 1, sizeof(spinor));
    if (tmp == NULL ) {
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      free(dirac3);
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
      free(rnd_vector);
      free(eigenvector);
      free(dirac0);
      free(dirac1);
      free(dirac2);
      free(dirac3);
      free(even);
      fprintf(stderr, "Could not allocate spinor!\nAborting...\n");
      exit(-1);
    }
#if (defined SSE || defined SSE2 || defined SSE3)
    odd = (spinor*) (((unsigned long int) (tmp) + ALIGN_BASE) & ~ALIGN_BASE);
#else
    odd = tmp;
#endif

#ifdef OMP
#pragma omp for private(tslice)
#endif
    for (tslice = 0; tslice < dilution_list[nr_dilution].size[0]; tslice++) {
      for (vec = 0; vec < vs; vec++) {
        zero_spinor_field(dirac0, VOLUMEPLUSRAND);
        zero_spinor_field(dirac1, VOLUMEPLUSRAND);
        zero_spinor_field(dirac2, VOLUMEPLUSRAND);
        zero_spinor_field(dirac3, VOLUMEPLUSRAND);

        for (v = vec * vs; v < (vec + 1) * vs; v++) {
          for (t = tslice; t < T; t += dilution_list[nr_dilution].size[0]) {
            // read in eigenvector and distribute it to the sources
            sprintf(filename, "./eigenvector.%03d.%03d.%04d", v, t, nr_conf);
#if DEBUG
#ifdef OMP
            printf("thread %d reading file %s\n", tid, filename);
            fflush(stdout);
#else
            printf("reading file %s\n", filename);
#endif
#endif
            read_su3_vector(eigenvector, filename, 0, t, 1);
            index = t * no_eigenvalues * 4 + v * 4;
            for (point = 0; point < block; point++) {
              _vector_add_mul( dirac0[block*t + point].s0, rnd_vector[index+0],
                  eigenvector[point]);
              _vector_add_mul( dirac1[block*t + point].s1, rnd_vector[index+1],
                  eigenvector[point]);
              _vector_add_mul( dirac2[block*t + point].s2, rnd_vector[index+2],
                  eigenvector[point]);
              _vector_add_mul( dirac3[block*t + point].s3, rnd_vector[index+3],
                  eigenvector[point]);
            }
          }
        }

        // write spinor field with entries at dirac 0
        convert_lexic_to_eo(even, odd, dirac0);
        sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 1
        convert_lexic_to_eo(even, odd, dirac1);
        sprintf(filename, "%s.%04d.%02d.%02d", "source1", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 2
        convert_lexic_to_eo(even, odd, dirac2);
        sprintf(filename, "%s.%04d.%02d.%02d", "source2", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
        // write spinor field with entries at dirac 3
        convert_lexic_to_eo(even, odd, dirac3);
        sprintf(filename, "%s.%04d.%02d.%02d", "source3", nr_conf, tslice, vec);
#if DEBUG
#ifdef OMP
        printf("thread %d writing file %s\n", tid, filename);
#else
        printf("writing file %s\n", filename);
#endif
#endif
        construct_writer(&writer, filename, 0);
        status = write_spinor(writer, &even, &odd, 1, 64);
        destruct_writer(writer);
      }

      create_input_files(4, tslice, nr_conf, nr_dilution, tid);
      for (j = 0; j < 4; j++) {
        sprintf(call, "%s -f dirac%d.%d-cg.input 1> /dev/null", inverterpath, j,
            tid);
#if DEBUG
#ifdef OMP
        printf("\n\nthread %d trying: %s for conf %d, t %d (full)\n", tid, call,
            nr_conf, tslice);
        fflush(stdout);
#else
        printf("\n\ntrying: %s for conf %d, t %d (full)\n", call,
            nr_conf, tslice);
#endif
#endif
        fflush(stdout);
        system(call);

      }
    }
    free(eigenvector);
    free(dirac0);
    free(dirac1);
    free(dirac2);
    free(dirac3);
    free(even);
    free(odd);
#ifdef OMP
  } // end of OMP region
#endif
  free(rnd_vector);
  return;
}

void add_dilution(const int d_type_t, const int d_type_d, const int d_type_l,
    const int d_t, const int d_d, const int d_l, const int d_seed,
    const int quark_type, const int smearing) {
  dilution * dptr = &dilution_list[no_dilution];
  if (no_dilution == max_no_dilution) {
    fprintf(stderr, "maximal number of dilutions (%d) exceeded!\n",
        max_no_dilution);
    exit(-1);
  }

  dptr->type[0] = d_type_t;
  dptr->type[1] = d_type_d;
  dptr->type[2] = d_type_l;
  dptr->size[2] = d_l;
  dptr->seed = d_seed;

  if (dptr->type[0] == D_INTER || dptr->type[0] == D_BLOCK) {
    if (d_t > T || d_t <= 0) {
      dptr->size[0] = T;
    } else {
      if (T % d_t != 0) {
        fprintf(stderr,
            "Dilution size is no divisor of time size\n Aborting...\n");
        exit(-1);
      }
      dptr->size[0] = d_t;
    }
  } else if (dptr->type[0] == D_FULL || dptr->type[0] == D_NONE) {
    dptr->size[0] = -1;
  } else {
    fprintf(stderr, "Dilution scheme for time not recognized!\nAborting...\n");
    exit(-2);
  }

  if (dptr->type[1] == D_INTER || dptr->type[1] == D_BLOCK) {
    fprintf(stderr,
        "Only full and no dilution in spin space implemented!\nSwitching to full dilution...\n");
    dptr->type[1] = D_FULL;
    dptr->size[1] = 4;
  } else if (dptr->type[1] == D_FULL) {
    dptr->size[1] = 4;
  } else if (dptr->type[1] == D_NONE) {
    dptr->size[1] = 1;
  } else {
    fprintf(stderr, "Dilution scheme for spin not recognized!\nAborting...\n");
    exit(-2);
  }

  if (dptr->type[2] == D_INTER || dptr->type[2] == D_BLOCK) {
    if (d_l > no_eigenvalues || d_l <= 0) {
      dptr->size[2] = no_eigenvalues;
    } else {
      if (no_eigenvalues % d_l != 0) {
        fprintf(stderr,
            "Dilution size is no divisor of LapH size\n Aborting...\n");
        exit(-1);
      }
      dptr->size[2] = d_l;
    }
  } else if (dptr->type[2] == D_FULL || dptr->type[2] == D_NONE) {
    dptr->size[2] = -1;
  } else {
    fprintf(stderr, "Dilution scheme for LapH not recognized!\nAborting...\n");
    exit(-2);
  }

  if (quark_type != D_UP && quark_type != D_DOWN) {
    fprintf(stderr, "Quark type not recognized!\nAborting...\n");
    exit(-2);
  } else {
    dptr->quark = quark_type;
  }

  if (smearing == D_STOCH || smearing == D_LOCAL) {
    dptr->smearing = smearing;
  } else {
    fprintf(stderr, "Smearing type not recognized!\nAborting...\n");
    exit(-2);
  }

  no_dilution++;
}
