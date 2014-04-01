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

// only needed for read_binary_eigenvector
#include <stdio.h>

#define BINARYINPUT 1
// end for read_binary_eigenvector

#define DEBUG 1
#define EIGENSYSTEMPATH "../"

int g_stochastical_run = 1;
int g_gpu_flag = 1;
int no_dilution = 0;
dilution dilution_list[max_no_dilution];

static int read_binary_eigenvector(su3_vector * const s, char * filename,
    int nr_eigenvector) {
  FILE *infile = fopen(filename, "rb");
  if (infile == NULL ) {
    fprintf(stderr, "Unable to find file %s.\nReturning...\n", filename);
    return -1;
  }
  fseek(infile, SPACEVOLUME * nr_eigenvector * sizeof(su3_vector), SEEK_SET);
  fread(s, sizeof(su3_vector), SPACEVOLUME, infile);

  fclose(infile);
  return 0;
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

void create_input_files(int const dirac, int const timeslice, int const conf,
    int const nr_dilution, int const thread) {
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
      if (dilution_list[nr_dilution].type[2] == D_FULL
          || dilution_list[nr_dilution].type[2] == D_INTER
          || dilution_list[nr_dilution].type[2] == D_BLOCK) {
        fprintf(file, "Indices = 0-%d\n\n",
            dilution_list[nr_dilution].size[2] - 1);
      } else if (dilution_list[nr_dilution].type[2] == D_NONE) {
        fprintf(file, "Indices = 0\n\n");
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
      fprintf(file, "  2kappamu = %f\n", operator_list[n_op].mu);
      fprintf(file, "  kappa = %f\n", operator_list[n_op].kappa);
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
      fprintf(file, "EndOperator\n\n");
    }
    if (g_gpu_flag) {
      fprintf(file, "BeginGPUInit\n");
      fprintf(file, "  MaxInnerSolverIteration = 1000\n");
      fprintf(file, "  InnerSolverPrecision = 1.0e-4\n");
      fprintf(file, "EndGPUInit\n");
    }

    fflush(file);
    fclose(file);
  }
  return;
}

void create_source_ti_df_li(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  int tslice = 0, vec = 0, point = 0;
  int status = 0, rnd_vec_size = T * 4 * no_eigenvalues;
  int count = 0, v = 0, t = 0;
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *dirac0 = NULL;
  spinor *dirac1 = NULL;
  spinor *dirac2 = NULL;
  spinor *dirac3 = NULL;
  WRITER* writer = NULL;
  _Complex double *rnd_vector = NULL;
  FILE* file;
  int index = 0;

  // generate random vector
  rnd_vector = (_Complex double*) calloc(rnd_vec_size, sizeof(_Complex double));
  if (rnd_vector == NULL ) {
    fprintf(stderr, "Could not allocate random vector!\nAborting...\n");
    exit(-1);
  }
  rnd_z2_vector(rnd_vector, rnd_vec_size);
  sprintf(filename, "randomvector.%03d.%s.T%s.%04d", nr_dilution,
      dilution_list[nr_dilution].quarktype,
      dilution_list[nr_dilution].typestring, nr_conf);
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
//  allocate spinors and eigenvectors
  eigenvector = (su3_vector*) calloc(SPACEVOLUME, sizeof(su3_vector));
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
#if BINARYINPUT
          sprintf(filename, "%seigenvectors.%04d.%03d", EIGENSYSTEMPATH,
              nr_conf, t);
#else
          sprintf(filename, "%seigenvector.%03d.%03d.%04d", EIGENSYSTEMPATH, v,
              t, nr_conf);
#endif
#if DEBUG
          printf("reading file %s\n", filename);
#endif
#if BINARYINPUT
          read_binary_eigenvector(eigenvector, filename, v);
#else
          read_su3_vector(eigenvector, filename, 0, t, 1);
#endif
          index = t * no_eigenvalues * 4 + v * 4;
          for (point = 0; point < SPACEVOLUME; point++) {
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s0,
                rnd_vector[index+0], eigenvector[point]);
            _vector_add_mul( dirac1[SPACEVOLUME*t + point].s1,
                rnd_vector[index+1], eigenvector[point]);
            _vector_add_mul( dirac2[SPACEVOLUME*t + point].s2,
                rnd_vector[index+2], eigenvector[point]);
            _vector_add_mul( dirac3[SPACEVOLUME*t + point].s3,
                rnd_vector[index+3], eigenvector[point]);
          }
        }
      }

      // write spinor field with entries at dirac 0
      sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac0, NULL, 1, 64);
      destruct_writer(writer);

//#if DEBUG
//      // write spinor field with entries at dirac 0
//      sprintf(filename, "%s.%04d.%02d.%02d", "source0_ascii", nr_conf, tslice,
//          vec);
//      printf("writing file %s\n", filename);
//      file = fopen(filename, "wb");
//      fwrite(dirac0, VOLUME, sizeof(spinor), file);
//      fclose(file);
//#endif

// write spinor field with entries at dirac 1
      sprintf(filename, "%s.%04d.%02d.%02d", "source1", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac1, NULL, 1, 64);
      destruct_writer(writer);

//#if DEBUG
//      // write spinor field with entries at dirac 0
//      sprintf(filename, "%s.%04d.%02d.%02d", "source1_ascii", nr_conf, tslice,
//          vec);
//      printf("writing file %s\n", filename);
//      file = fopen(filename, "wb");
//      fwrite(dirac1, VOLUME, sizeof(spinor), file);
//      fclose(file);
//#endif

// write spinor field with entries at dirac 2
      sprintf(filename, "%s.%04d.%02d.%02d", "source2", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac2, NULL, 1, 64);
      destruct_writer(writer);

//#if DEBUG
//      // write spinor field with entries at dirac 0
//      sprintf(filename, "%s.%04d.%02d.%02d", "source2_ascii", nr_conf, tslice,
//          vec);
//      printf("writing file %s\n", filename);
//      file = fopen(filename, "wb");
//      fwrite(dirac2, VOLUME, sizeof(spinor), file);
//      fclose(file);
//#endif

// write spinor field with entries at dirac 3
      sprintf(filename, "%s.%04d.%02d.%02d", "source3", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac3, NULL, 1, 64);
      destruct_writer(writer);

//#if DEBUG
//      // write spinor field with entries at dirac 0
//      sprintf(filename, "%s.%04d.%02d.%02d", "source3_ascii", nr_conf, tslice,
//          vec);
//      printf("writing file %s\n", filename);
//      file = fopen(filename, "wb");
//      fwrite(dirac3, VOLUME, sizeof(spinor), file);
//      fclose(file);
//#endif
    }
  }
  free(eigenvector);
  free(dirac0);
  free(dirac1);
  free(dirac2);
  free(dirac3);
  free(rnd_vector);
  return;
}

void create_source_ti_df_lb(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  int tslice = 0, vec = 0, point = 0;
  int status = 0, rnd_vec_size = T * 4 * no_eigenvalues;
  int count = 0, v = 0, t = 0;
  int vs = dilution_list[nr_dilution].size[2];
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
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
  sprintf(filename, "randomvector.%03d.%s.T%s.%04d", nr_dilution,
      dilution_list[nr_dilution].quarktype,
      dilution_list[nr_dilution].typestring, nr_conf);
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
//  allocate spinors and eigenvectors
  eigenvector = (su3_vector*) calloc(SPACEVOLUME, sizeof(su3_vector));
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

  for (tslice = 0; tslice < dilution_list[nr_dilution].size[0]; tslice++) {
    for (vec = 0; vec < (no_eigenvalues / vs); vec++) {
      zero_spinor_field(dirac0, VOLUMEPLUSRAND);
      zero_spinor_field(dirac1, VOLUMEPLUSRAND);
      zero_spinor_field(dirac2, VOLUMEPLUSRAND);
      zero_spinor_field(dirac3, VOLUMEPLUSRAND);

      for (v = vec * vs; v < (vec + 1) * vs; v++) {
        for (t = tslice; t < T; t += dilution_list[nr_dilution].size[0]) {
          // read in eigenvector and distribute it to the sources
#if BINARYINPUT
          sprintf(filename, "%seigenvectors.%04d.%03d", EIGENSYSTEMPATH,
              nr_conf, t);
#else
          sprintf(filename, "%seigenvector.%03d.%03d.%04d", EIGENSYSTEMPATH, v,
              t, nr_conf);
#endif
#if DEBUG
          printf("reading file %s\n", filename);
#endif
#if BINARYINPUT
          read_binary_eigenvector(eigenvector, filename, v);
#else
          read_su3_vector(eigenvector, filename, 0, t, 1);
#endif
          index = t * no_eigenvalues * 4 + v * 4;
          for (point = 0; point < SPACEVOLUME; point++) {
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s0,
                rnd_vector[index+0], eigenvector[point]);
            _vector_add_mul( dirac1[SPACEVOLUME*t + point].s1,
                rnd_vector[index+1], eigenvector[point]);
            _vector_add_mul( dirac2[SPACEVOLUME*t + point].s2,
                rnd_vector[index+2], eigenvector[point]);
            _vector_add_mul( dirac3[SPACEVOLUME*t + point].s3,
                rnd_vector[index+3], eigenvector[point]);
          }
        }
      }

      // write spinor field with entries at dirac 0
      sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac0, NULL, 1, 64);
      destruct_writer(writer);
      // write spinor field with entries at dirac 1
      sprintf(filename, "%s.%04d.%02d.%02d", "source1", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac1, NULL, 1, 64);
      destruct_writer(writer);
      // write spinor field with entries at dirac 2
      sprintf(filename, "%s.%04d.%02d.%02d", "source2", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac2, NULL, 1, 64);
      destruct_writer(writer);
      // write spinor field with entries at dirac 3
      sprintf(filename, "%s.%04d.%02d.%02d", "source3", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac3, NULL, 1, 64);
      destruct_writer(writer);
    }
  }
  free(eigenvector);
  free(dirac0);
  free(dirac1);
  free(dirac2);
  free(dirac3);
  free(rnd_vector);
  return;
}

void create_source_tb_df_li(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  int tslice = 0, vec = 0, point = 0;
  int status = 0, rnd_vec_size = T * 4 * no_eigenvalues;
  int count = 0, v = 0, t = 0, ts = dilution_list[nr_dilution].size[0], vs =
      dilution_list[nr_dilution].size[2];
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
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
  sprintf(filename, "randomvector.%03d.%s.T%s.%04d", nr_dilution,
      dilution_list[nr_dilution].quarktype,
      dilution_list[nr_dilution].typestring, nr_conf);
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
//  allocate spinors and eigenvectors
  eigenvector = (su3_vector*) calloc(SPACEVOLUME, sizeof(su3_vector));
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

  for (tslice = 0; tslice < (T / ts); tslice++) {
    for (vec = 0; vec < vs; vec++) {
      zero_spinor_field(dirac0, VOLUMEPLUSRAND);
      zero_spinor_field(dirac1, VOLUMEPLUSRAND);
      zero_spinor_field(dirac2, VOLUMEPLUSRAND);
      zero_spinor_field(dirac3, VOLUMEPLUSRAND);

      for (t = tslice * ts; t < (tslice + 1) * ts; t++) {
        for (v = vec; v < no_eigenvalues; v += vs) {
          // read in eigenvector and distribute it to the sources
          sprintf(filename, "./eigenvector.%03d.%03d.%04d", vec, t, nr_conf);
#if BINARYINPUT
          sprintf(filename, "%seigenvectors.%04d.%03d", EIGENSYSTEMPATH,
              nr_conf, t);
#else
          sprintf(filename, "%seigenvector.%03d.%03d.%04d", EIGENSYSTEMPATH, v,
              t, nr_conf);
#endif
#if DEBUG
          printf("reading file %s\n", filename);
#endif
#if BINARYINPUT
          read_binary_eigenvector(eigenvector, filename, v);
#else
          read_su3_vector(eigenvector, filename, 0, t, 1);
#endif
          index = t * no_eigenvalues * 4 + v * 4;
          for (point = 0; point < SPACEVOLUME; point++) {
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s0,
                rnd_vector[index+0], eigenvector[point]);
            _vector_add_mul( dirac1[SPACEVOLUME*t + point].s1,
                rnd_vector[index+1], eigenvector[point]);
            _vector_add_mul( dirac2[SPACEVOLUME*t + point].s2,
                rnd_vector[index+2], eigenvector[point]);
            _vector_add_mul( dirac3[SPACEVOLUME*t + point].s3,
                rnd_vector[index+3], eigenvector[point]);
          }
        }
      }

      // write spinor field with entries at dirac 0
      sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac0, NULL, 1, 64);
      destruct_writer(writer);

//#if DEBUG
//      // write spinor field with entries at dirac 0
//      sprintf(filename, "%s.%04d.%02d.%02d", "source0_ascii", nr_conf, tslice,
//          vec);
//      printf("writing file %s\n", filename);
//      file = fopen(filename, "wb");
//      fwrite(dirac0, VOLUME, sizeof(spinor), file);
//      fclose(file);
//#endif

// write spinor field with entries at dirac 1
      sprintf(filename, "%s.%04d.%02d.%02d", "source1", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac1, NULL, 1, 64);
      destruct_writer(writer);

//#if DEBUG
//      // write spinor field with entries at dirac 0
//      sprintf(filename, "%s.%04d.%02d.%02d", "source1_ascii", nr_conf, tslice,
//          vec);
//      printf("writing file %s\n", filename);
//      file = fopen(filename, "wb");
//      fwrite(dirac1, VOLUME, sizeof(spinor), file);
//      fclose(file);
//#endif

// write spinor field with entries at dirac 2
      sprintf(filename, "%s.%04d.%02d.%02d", "source2", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac2, NULL, 1, 64);
      destruct_writer(writer);

//#if DEBUG
//      // write spinor field with entries at dirac 0
//      sprintf(filename, "%s.%04d.%02d.%02d", "source2_ascii", nr_conf, tslice,
//          vec);
//      printf("writing file %s\n", filename);
//      file = fopen(filename, "wb");
//      fwrite(dirac2, VOLUME, sizeof(spinor), file);
//      fclose(file);
//#endif

// write spinor field with entries at dirac 3
      sprintf(filename, "%s.%04d.%02d.%02d", "source3", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac3, NULL, 1, 64);
      destruct_writer(writer);

//#if DEBUG
//      // write spinor field with entries at dirac 0
//      sprintf(filename, "%s.%04d.%02d.%02d", "source3_ascii", nr_conf, tslice,
//          vec);
//      printf("writing file %s\n", filename);
//      file = fopen(filename, "wb");
//      fwrite(dirac3, VOLUME, sizeof(spinor), file);
//      fclose(file);
//#endif
    }
  }
  free(eigenvector);
  free(dirac0);
  free(dirac1);
  free(dirac2);
  free(dirac3);
  free(rnd_vector);
  return;
}

void create_source_tb_df_lb(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  int tslice = 0, vec = 0, point = 0;
  int status = 0, rnd_vec_size = T * 4 * no_eigenvalues;
  int count = 0, v = 0, t = 0, ts = dilution_list[nr_dilution].size[0], vs =
      dilution_list[nr_dilution].size[2];
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
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
  sprintf(filename, "randomvector.%03d.%s.T%s.%04d", nr_dilution,
      dilution_list[nr_dilution].quarktype,
      dilution_list[nr_dilution].typestring, nr_conf);
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
//  allocate spinors and eigenvectors
  eigenvector = (su3_vector*) calloc(SPACEVOLUME, sizeof(su3_vector));
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

  for (tslice = 0; tslice < (T / ts); tslice++) {
    for (vec = 0; vec < (no_eigenvalues / vs); vec++) {
      zero_spinor_field(dirac0, VOLUMEPLUSRAND);
      zero_spinor_field(dirac1, VOLUMEPLUSRAND);
      zero_spinor_field(dirac2, VOLUMEPLUSRAND);
      zero_spinor_field(dirac3, VOLUMEPLUSRAND);

      for (t = tslice * ts; t < (tslice + 1) * ts; t++) {
        for (v = vec * vs; v < (vec + 1) * vs; v++) {
          // read in eigenvector and distribute it to the sources
          sprintf(filename, "./eigenvector.%03d.%03d.%04d", vec, t, nr_conf);
#if BINARYINPUT
          sprintf(filename, "%seigenvectors.%04d.%03d", EIGENSYSTEMPATH,
              nr_conf, t);
#else
          sprintf(filename, "%seigenvector.%03d.%03d.%04d", EIGENSYSTEMPATH, v,
              t, nr_conf);
#endif
#if DEBUG
          printf("reading file %s\n", filename);
#endif
#if BINARYINPUT
          read_binary_eigenvector(eigenvector, filename, v);
#else
          read_su3_vector(eigenvector, filename, 0, t, 1);
#endif
          index = t * no_eigenvalues * 4 + v * 4;
          for (point = 0; point < SPACEVOLUME; point++) {
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s0,
                rnd_vector[index+0], eigenvector[point]);
            _vector_add_mul( dirac1[SPACEVOLUME*t + point].s1,
                rnd_vector[index+1], eigenvector[point]);
            _vector_add_mul( dirac2[SPACEVOLUME*t + point].s2,
                rnd_vector[index+2], eigenvector[point]);
            _vector_add_mul( dirac3[SPACEVOLUME*t + point].s3,
                rnd_vector[index+3], eigenvector[point]);
          }
        }
      }

      // write spinor field with entries at dirac 0
      sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac0, NULL, 1, 64);
      destruct_writer(writer);
      // write spinor field with entries at dirac 1
      sprintf(filename, "%s.%04d.%02d.%02d", "source1", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac1, NULL, 1, 64);
      destruct_writer(writer);
      // write spinor field with entries at dirac 2
      sprintf(filename, "%s.%04d.%02d.%02d", "source2", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac2, NULL, 1, 64);
      destruct_writer(writer);
      // write spinor field with entries at dirac 3
      sprintf(filename, "%s.%04d.%02d.%02d", "source3", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac3, NULL, 1, 64);
      destruct_writer(writer);
    }
  }
  free(eigenvector);
  free(dirac0);
  free(dirac1);
  free(dirac2);
  free(dirac3);
  free(rnd_vector);
  return;
}

void create_source_ti_dn_li(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  int tslice = 0, vec = 0, point = 0;
  int status = 0, rnd_vec_size = T * 4 * no_eigenvalues;
  int count = 0, v = 0, t = 0;
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *dirac0 = NULL;
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
  sprintf(filename, "randomvector.%03d.%s.T%s.%04d", nr_dilution,
      dilution_list[nr_dilution].quarktype,
      dilution_list[nr_dilution].typestring, nr_conf);
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
//  allocate spinors and eigenvectors
  eigenvector = (su3_vector*) calloc(SPACEVOLUME, sizeof(su3_vector));
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

  for (tslice = 0; tslice < dilution_list[nr_dilution].size[0]; tslice++) {
    for (vec = 0; vec < dilution_list[nr_dilution].size[2]; vec++) {
      zero_spinor_field(dirac0, VOLUMEPLUSRAND);

      for (v = vec; v < no_eigenvalues; v +=
          dilution_list[nr_dilution].size[2]) {
        for (t = tslice; t < T; t += dilution_list[nr_dilution].size[0]) {
          // read in eigenvector and distribute it to the sources
          sprintf(filename, "./eigenvector.%03d.%03d.%04d", v, t, nr_conf);
#if BINARYINPUT
          sprintf(filename, "%seigenvectors.%04d.%03d", EIGENSYSTEMPATH,
              nr_conf, t);
#else
          sprintf(filename, "%seigenvector.%03d.%03d.%04d", EIGENSYSTEMPATH, v,
              t, nr_conf);
#endif
#if DEBUG
          printf("reading file %s\n", filename);
#endif
#if BINARYINPUT
          read_binary_eigenvector(eigenvector, filename, v);
#else
          read_su3_vector(eigenvector, filename, 0, t, 1);
#endif
          index = t * no_eigenvalues * 4 + v * 4;
          for (point = 0; point < SPACEVOLUME; point++) {
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s0,
                rnd_vector[index+0], eigenvector[point]);
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s1,
                rnd_vector[index+1], eigenvector[point]);
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s2,
                rnd_vector[index+2], eigenvector[point]);
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s3,
                rnd_vector[index+3], eigenvector[point]);
          }
        }
      }

      // write spinor field with entries at dirac 0
      sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac0, NULL, 1, 64);
      destruct_writer(writer);
//#if DEBUG
//      // write spinor field with entries at dirac 0
//      sprintf(filename, "%s.%04d.%02d.%02d", "source0_ascii", nr_conf, tslice,
//          vec);
//      printf("writing file %s\n", filename);
//      file = fopen(filename, "wb");
//      fwrite(dirac0, VOLUME, sizeof(spinor), file);
//      fclose(file);
//#endif
    }
  }
  free(eigenvector);
  free(dirac0);
  free(rnd_vector);
  return;
}

void create_source_ti_dn_lb(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  int tslice = 0, vec = 0, point = 0;
  int status = 0, rnd_vec_size = T * 4 * no_eigenvalues;
  int count = 0, v = 0, t = 0;
  int vs = dilution_list[nr_dilution].size[2];
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *dirac0 = NULL;
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
  sprintf(filename, "randomvector.%03d.%s.T%s.%04d", nr_dilution,
      dilution_list[nr_dilution].quarktype,
      dilution_list[nr_dilution].typestring, nr_conf);
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
//  allocate spinors and eigenvectors
  eigenvector = (su3_vector*) calloc(SPACEVOLUME, sizeof(su3_vector));
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

  for (tslice = 0; tslice < dilution_list[nr_dilution].size[0]; tslice++) {
    for (vec = 0; vec < (no_eigenvalues / vs); vec++) {
      zero_spinor_field(dirac0, VOLUMEPLUSRAND);

      for (v = vec * vs; v < (vec + 1) * vs; v++) {
        for (t = tslice; t < T; t += dilution_list[nr_dilution].size[0]) {
          // read in eigenvector and distribute it to the sources
          sprintf(filename, "./eigenvector.%03d.%03d.%04d", v, t, nr_conf);
#if BINARYINPUT
          sprintf(filename, "%seigenvectors.%04d.%03d", EIGENSYSTEMPATH,
              nr_conf, t);
#else
          sprintf(filename, "%seigenvector.%03d.%03d.%04d", EIGENSYSTEMPATH, v,
              t, nr_conf);
#endif
#if DEBUG
          printf("reading file %s\n", filename);
#endif
#if BINARYINPUT
          read_binary_eigenvector(eigenvector, filename, v);
#else
          read_su3_vector(eigenvector, filename, 0, t, 1);
#endif
          index = t * no_eigenvalues * 4 + v * 4;
          for (point = 0; point < SPACEVOLUME; point++) {
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s0,
                rnd_vector[index+0], eigenvector[point]);
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s1,
                rnd_vector[index+1], eigenvector[point]);
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s2,
                rnd_vector[index+2], eigenvector[point]);
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s3,
                rnd_vector[index+3], eigenvector[point]);
          }
        }
      }

      // write spinor field with entries at dirac 0
      sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac0, NULL, 1, 64);
      destruct_writer(writer);
    }
  }
  free(eigenvector);
  free(dirac0);
  free(rnd_vector);
  return;
}

void create_source_tb_dn_li(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  int tslice = 0, vec = 0, point = 0;
  int status = 0, rnd_vec_size = T * 4 * no_eigenvalues;
  int count = 0, v = 0, t = 0, ts = dilution_list[nr_dilution].size[0], vs =
      dilution_list[nr_dilution].size[2];
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *dirac0 = NULL;
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
  sprintf(filename, "randomvector.%03d.%s.T%s.%04d", nr_dilution,
      dilution_list[nr_dilution].quarktype,
      dilution_list[nr_dilution].typestring, nr_conf);
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
//  allocate spinors and eigenvectors
  eigenvector = (su3_vector*) calloc(SPACEVOLUME, sizeof(su3_vector));
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

  for (tslice = 0; tslice < (T / ts); tslice++) {
    for (vec = 0; vec < vs; vec++) {
      zero_spinor_field(dirac0, VOLUMEPLUSRAND);

      for (t = tslice * ts; t < (tslice + 1) * ts; t++) {
        for (v = vec; v < no_eigenvalues; v += vs) {
          // read in eigenvector and distribute it to the sources
          sprintf(filename, "./eigenvector.%03d.%03d.%04d", vec, t, nr_conf);
#if BINARYINPUT
          sprintf(filename, "%seigenvectors.%04d.%03d", EIGENSYSTEMPATH,
              nr_conf, t);
#else
          sprintf(filename, "%seigenvector.%03d.%03d.%04d", EIGENSYSTEMPATH, v,
              t, nr_conf);
#endif
#if DEBUG
          printf("reading file %s\n", filename);
#endif
#if BINARYINPUT
          read_binary_eigenvector(eigenvector, filename, v);
#else
          read_su3_vector(eigenvector, filename, 0, t, 1);
#endif
          index = t * no_eigenvalues * 4 + v * 4;
          for (point = 0; point < SPACEVOLUME; point++) {
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s0,
                rnd_vector[index+0], eigenvector[point]);
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s1,
                rnd_vector[index+1], eigenvector[point]);
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s2,
                rnd_vector[index+2], eigenvector[point]);
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s3,
                rnd_vector[index+3], eigenvector[point]);
          }
        }
      }

      // write spinor field with entries at dirac 0
      sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac0, NULL, 1, 64);
      destruct_writer(writer);
    }
  }
  free(eigenvector);
  free(dirac0);
  free(rnd_vector);
  return;
}

void create_source_tb_dn_lb(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  int tslice = 0, vec = 0, point = 0;
  int status = 0, rnd_vec_size = T * 4 * no_eigenvalues;
  int count = 0, v = 0, t = 0, ts = dilution_list[nr_dilution].size[0], vs =
      dilution_list[nr_dilution].size[2];
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *dirac0 = NULL;
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
  sprintf(filename, "randomvector.%03d.%s.T%s.%04d", nr_dilution,
      dilution_list[nr_dilution].quarktype,
      dilution_list[nr_dilution].typestring, nr_conf);
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
//  allocate spinors and eigenvectors
  eigenvector = (su3_vector*) calloc(SPACEVOLUME, sizeof(su3_vector));
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

  for (tslice = 0; tslice < (T / ts); tslice++) {
    for (vec = 0; vec < (no_eigenvalues / vs); vec++) {
      zero_spinor_field(dirac0, VOLUMEPLUSRAND);

      for (t = tslice * ts; t < (tslice + 1) * ts; t++) {
        for (v = vec * vs; v < (vec + 1) * vs; v++) {
          // read in eigenvector and distribute it to the sources
          sprintf(filename, "./eigenvector.%03d.%03d.%04d", vec, t, nr_conf);
#if BINARYINPUT
          sprintf(filename, "%seigenvectors.%04d.%03d", EIGENSYSTEMPATH,
              nr_conf, t);
#else
          sprintf(filename, "%seigenvector.%03d.%03d.%04d", EIGENSYSTEMPATH, v,
              t, nr_conf);
#endif
#if DEBUG
          printf("reading file %s\n", filename);
#endif
#if BINARYINPUT
          read_binary_eigenvector(eigenvector, filename, v);
#else
          read_su3_vector(eigenvector, filename, 0, t, 1);
#endif
          index = t * no_eigenvalues * 4 + v * 4;
          for (point = 0; point < SPACEVOLUME; point++) {
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s0,
                rnd_vector[index+0], eigenvector[point]);
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s1,
                rnd_vector[index+1], eigenvector[point]);
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s2,
                rnd_vector[index+2], eigenvector[point]);
            _vector_add_mul( dirac0[SPACEVOLUME*t + point].s3,
                rnd_vector[index+3], eigenvector[point]);
          }
        }
      }

      // write spinor field with entries at dirac 0
      sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac0, NULL, 1, 64);
      destruct_writer(writer);
    }
  }
  free(eigenvector);
  free(dirac0);
  free(rnd_vector);
  return;
}

void create_source_t1_df_lf(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  int vec = 0, point = 0, j = 0;
  int status = 0, count = 0;
  int rnd_vec_size = 4 * no_eigenvalues;
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *dirac0 = NULL;
  spinor *dirac1 = NULL;
  spinor *dirac2 = NULL;
  spinor *dirac3 = NULL;
  WRITER* writer = NULL;
  _Complex double *rnd_vector = NULL;
  FILE* file;
  int index = 0;

  // allocate random vector
  rnd_vector = (_Complex double*) calloc(rnd_vec_size, sizeof(_Complex double));
  if (rnd_vector == NULL ) {
    fprintf(stderr, "Could not allocate random vector!\nAborting...\n");
    exit(-1);
  }
  // fill random vector
  rnd_z2_vector(rnd_vector, rnd_vec_size);
  //for testing purposes
  for (j = 0; j < rnd_vec_size; j++) {
    rnd_vector[j] = 1.;
  }
  // write random vector to file
  sprintf(filename, "randomvector.%03d.%s.T%s.%04d", nr_dilution,
      dilution_list[nr_dilution].quarktype,
      dilution_list[nr_dilution].typestring, nr_conf);
#if DEBUG
  printf("\nwriting random vector to file %s\n", filename);
#endif
  if ((file = fopen(filename, "wb")) == NULL ) {
    fprintf(stderr, "Could not open file %s for random vector\nAborting...\n",
        filename);
    exit(-1);
  }
  count = fwrite(rnd_vector, sizeof(_Complex double), rnd_vec_size, file);
  // check that complete random vector was written
  if (count != rnd_vec_size) {
    fprintf(stderr,
        "Could not print all data to file %s for random vector (%d of %d)\nAborting...\n",
        filename, count, rnd_vec_size);
    exit(-1);
  }
  fclose(file);

//  allocate spinors and eigenvectors
  eigenvector = (su3_vector*) calloc(SPACEVOLUME, sizeof(su3_vector));
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

  // main loop over eigenvectors
  for (vec = 0; vec < no_eigenvalues; vec++) {
    // zeroing of the spinor fields
    zero_spinor_field(dirac0, VOLUMEPLUSRAND);
    zero_spinor_field(dirac1, VOLUMEPLUSRAND);
    zero_spinor_field(dirac2, VOLUMEPLUSRAND);
    zero_spinor_field(dirac3, VOLUMEPLUSRAND);

    // read in eigenvector
#if BINARYINPUT
    sprintf(filename, "%seigenvectors.%04d.%03d", EIGENSYSTEMPATH, nr_conf, 0);
#else
    sprintf(filename, "%seigenvector.%03d.%03d.%04d", EIGENSYSTEMPATH, vec, 0,
        nr_conf);
#endif
#if DEBUG
    printf("reading file %s\n", filename);
#endif
#if BINARYINPUT
    read_binary_eigenvector(eigenvector, filename, 0);
#else
    read_su3_vector(eigenvector, filename, 0, 0, 1);
#endif
    // helper index for the random vector
    index = vec * 4;
    // multiplication of the eigenvector with the randomvector
    for (point = 0; point < SPACEVOLUME; point++) {
      _vector_mul( dirac0[point].s0, rnd_vector[index+0], eigenvector[point]);
      _vector_mul( dirac1[point].s1, rnd_vector[index+1], eigenvector[point]);
      _vector_mul( dirac2[point].s2, rnd_vector[index+2], eigenvector[point]);
      _vector_mul( dirac3[point].s3, rnd_vector[index+3], eigenvector[point]);
      // for testing purposes
//      dirac1[point].s1 = dirac0[point].s0;
//      dirac2[point].s2 = dirac0[point].s0;
//      dirac3[point].s3 = dirac0[point].s0;
    }

    // write spinor field dirac component 0
    sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, 0, vec);
#if DEBUG
    printf("writing file %s\n", filename);
#endif
    construct_writer(&writer, filename, 0);
    status = write_spinor(writer, &dirac0, NULL, 1, 64);
    destruct_writer(writer);

    // write spinor field dirac component 1
    sprintf(filename, "%s.%04d.%02d.%02d", "source1", nr_conf, 0, vec);
#if DEBUG
    printf("writing file %s\n", filename);
#endif
    construct_writer(&writer, filename, 0);
    status = write_spinor(writer, &dirac1, NULL, 1, 64);
    destruct_writer(writer);

    // write spinor field dirac component 0
    sprintf(filename, "%s.%04d.%02d.%02d", "source2", nr_conf, 0, vec);
#if DEBUG
    printf("writing file %s\n", filename);
#endif
    construct_writer(&writer, filename, 0);
    status = write_spinor(writer, &dirac2, NULL, 1, 64);
    destruct_writer(writer);

    // write spinor field dirac component 0
    sprintf(filename, "%s.%04d.%02d.%02d", "source3", nr_conf, 0, vec);
#if DEBUG
    printf("writing file %s\n", filename);
#endif
    construct_writer(&writer, filename, 0);
    status = write_spinor(writer, &dirac3, NULL, 1, 64);
    destruct_writer(writer);
  }

  free(eigenvector);
  free(dirac0);
  free(dirac1);
  free(dirac2);
  free(dirac3);
  free(rnd_vector);
  return;
}

void create_source_tbi2_df_li(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  int tslice = 0, tb = 0, ti = 0, t = 0, vec = 0, v = 0;
  int count = 0, status = 0, point = 0, index = 0;
  int rnd_vec_size = T * 4 * no_eigenvalues;
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *dirac0 = NULL;
  spinor *dirac1 = NULL;
  spinor *dirac2 = NULL;
  spinor *dirac3 = NULL;
  WRITER* writer = NULL;
  _Complex double *rnd_vector = NULL;
  FILE* file;

// allocate random vector
  rnd_vector = (_Complex double*) calloc(rnd_vec_size, sizeof(_Complex double));
  if (rnd_vector == NULL ) {
    fprintf(stderr, "Could not allocate random vector!\nAborting...\n");
    exit(-1);
  }
  rnd_z2_vector(rnd_vector, rnd_vec_size);
// filename for the random vector file
  sprintf(filename, "randomvector.%03d.%s.Tib.%04d", nr_dilution,
      dilution_list[nr_dilution].quarktype, nr_conf);
#if DEBUG
  printf("\nwriting random vector to file %s\n", filename);
#endif
// write random vector to file
  if ((file = fopen(filename, "wb")) == NULL ) {
    fprintf(stderr, "Could not open file %s for random vector\nAborting...\n",
        filename);
    exit(-1);
  }
  count = fwrite(rnd_vector, sizeof(_Complex double), rnd_vec_size, file);
// check the number of elements written is correct, otherwise fail!
  if (count != rnd_vec_size) {
    fprintf(stderr,
        "Could not print all data to file %s for random vector (%d of %d)\nAborting...\n",
        filename, count, rnd_vec_size);
    exit(-1);
  }
  fclose(file);
//  allocate spinors and eigenvectors
  eigenvector = (su3_vector*) calloc(SPACEVOLUME, sizeof(su3_vector));
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

// main calculation loop
// here, block-interlace dilution with length 2-2 is hard coded in time direction
// a sanity check for this is already done in the add_dilution routine
  for (tslice = 0; tslice < dilution_list[nr_dilution].size[0]; tslice++) {
    for (vec = 0; vec < dilution_list[nr_dilution].size[2]; vec++) {
      // zero the spinor fields
      zero_spinor_field(dirac0, VOLUMEPLUSRAND);
      zero_spinor_field(dirac1, VOLUMEPLUSRAND);
      zero_spinor_field(dirac2, VOLUMEPLUSRAND);
      zero_spinor_field(dirac3, VOLUMEPLUSRAND);

      // interlace loop in eigenvector space
      for (v = vec; v < no_eigenvalues; v +=
          dilution_list[nr_dilution].size[2]) {
        for (ti = tslice * 2; ti < T; ti += T / 2) {
          for (tb = -1; tb < 1; tb++) { // negative to get the times T-1 and 0 together!
            t = (ti + tb + T) % T;

            // read in eigenvector and distribute it to the sources
#if BINARYINPUT // binary eigenvector files
            sprintf(filename, "%seigenvectors.%04d.%03d", EIGENSYSTEMPATH,
                nr_conf, t);
#if DEBUG
            printf("reading file %s\n", filename);
#endif
            read_binary_eigenvector(eigenvector, filename, v);
#else  // ILDG format eigenvector files
            sprintf(filename, "%seigenvector.%03d.%03d.%04d", EIGENSYSTEMPATH,
                v, t, nr_conf);
#if DEBUG
            printf("reading file %s\n", filename);
#endif
            read_su3_vector(eigenvector, filename, 0, t, 1);
#endif

            // loop over position space
            index = t * no_eigenvalues * 4 + v * 4;
            for (point = 0; point < SPACEVOLUME; point++) {
              _vector_add_mul( dirac0[SPACEVOLUME*t + point].s0,
                  rnd_vector[index+0], eigenvector[point]);
              _vector_add_mul( dirac1[SPACEVOLUME*t + point].s1,
                  rnd_vector[index+1], eigenvector[point]);
              _vector_add_mul( dirac2[SPACEVOLUME*t + point].s2,
                  rnd_vector[index+2], eigenvector[point]);
              _vector_add_mul( dirac3[SPACEVOLUME*t + point].s3,
                  rnd_vector[index+3], eigenvector[point]);
            }
          }
        }
      }

      // write spinor field dirac 0
      sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac0, NULL, 1, 64);
      destruct_writer(writer);

      // write spinor field dirac 1
      sprintf(filename, "%s.%04d.%02d.%02d", "source1", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac1, NULL, 1, 64);
      destruct_writer(writer);

      // write spinor field dirac 2
      sprintf(filename, "%s.%04d.%02d.%02d", "source2", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac2, NULL, 1, 64);
      destruct_writer(writer);

      // write spinor field dirac 3
      sprintf(filename, "%s.%04d.%02d.%02d", "source3", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac3, NULL, 1, 64);
      destruct_writer(writer);
    }

  }
  free(eigenvector);
  free(dirac0);
  free(dirac1);
  free(dirac2);
  free(dirac3);
  free(rnd_vector);
  return;
}

void create_source_tbi3_df_li(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  int tslice = 0, tb = 0, ti = 0, t = 0, vec = 0, v = 0;
  int count = 0, status = 0, point = 0, index = 0;
  int rnd_vec_size = T * 4 * no_eigenvalues;
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *dirac0 = NULL;
  spinor *dirac1 = NULL;
  spinor *dirac2 = NULL;
  spinor *dirac3 = NULL;
  WRITER* writer = NULL;
  _Complex double *rnd_vector = NULL;
  FILE* file;

// allocate random vector
  rnd_vector = (_Complex double*) calloc(rnd_vec_size, sizeof(_Complex double));
  if (rnd_vector == NULL ) {
    fprintf(stderr, "Could not allocate random vector!\nAborting...\n");
    exit(-1);
  }
  rnd_z2_vector(rnd_vector, rnd_vec_size);
// filename for the random vector file
  sprintf(filename, "randomvector.%03d.%s.Tib.%04d", nr_dilution,
      dilution_list[nr_dilution].quarktype, nr_conf);
#if DEBUG
  printf("\nwriting random vector to file %s\n", filename);
#endif
// write random vector to file
  if ((file = fopen(filename, "wb")) == NULL ) {
    fprintf(stderr, "Could not open file %s for random vector\nAborting...\n",
        filename);
    exit(-1);
  }
  count = fwrite(rnd_vector, sizeof(_Complex double), rnd_vec_size, file);
// check the number of elements written is correct, otherwise fail!
  if (count != rnd_vec_size) {
    fprintf(stderr,
        "Could not print all data to file %s for random vector (%d of %d)\nAborting...\n",
        filename, count, rnd_vec_size);
    exit(-1);
  }
  fclose(file);
//  allocate spinors and eigenvectors
  eigenvector = (su3_vector*) calloc(SPACEVOLUME, sizeof(su3_vector));
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

// main calculation loop
// here, block-interlace dilution with length 3-3 is hard coded in time direction
// a sanity check for this is already done in the add_dilution routine
  for (tslice = 0; tslice < dilution_list[nr_dilution].size[0]; tslice++) {
    for (vec = 0; vec < dilution_list[nr_dilution].size[2]; vec++) {
      // zero the spinor fields
      zero_spinor_field(dirac0, VOLUMEPLUSRAND);
      zero_spinor_field(dirac1, VOLUMEPLUSRAND);
      zero_spinor_field(dirac2, VOLUMEPLUSRAND);
      zero_spinor_field(dirac3, VOLUMEPLUSRAND);

      // interlace loop in eigenvector space
      for (v = vec; v < no_eigenvalues; v +=
          dilution_list[nr_dilution].size[2]) {
        for (ti = tslice * 3; ti < T; ti += T / 2) {
          for (tb = -1; tb < 2; tb++) { // negative to get the times T-1 and 0 together!
            t = (ti + tb + T) % T;

            // read in eigenvector and distribute it to the sources
#if BINARYINPUT // binary eigenvector files
            sprintf(filename, "%seigenvectors.%04d.%03d", EIGENSYSTEMPATH,
                nr_conf, t);
#if DEBUG
            printf("reading file %s\n", filename);
#endif
            read_binary_eigenvector(eigenvector, filename, v);
#else  // ILDG format eigenvector files
            sprintf(filename, "%seigenvector.%03d.%03d.%04d", EIGENSYSTEMPATH,
                v, t, nr_conf);
#if DEBUG
            printf("reading file %s\n", filename);
#endif
            read_su3_vector(eigenvector, filename, 0, t, 1);
#endif

            // loop over position space
            index = t * no_eigenvalues * 4 + v * 4;
            for (point = 0; point < SPACEVOLUME; point++) {
              _vector_add_mul( dirac0[SPACEVOLUME*t + point].s0,
                  rnd_vector[index+0], eigenvector[point]);
              _vector_add_mul( dirac1[SPACEVOLUME*t + point].s1,
                  rnd_vector[index+1], eigenvector[point]);
              _vector_add_mul( dirac2[SPACEVOLUME*t + point].s2,
                  rnd_vector[index+2], eigenvector[point]);
              _vector_add_mul( dirac3[SPACEVOLUME*t + point].s3,
                  rnd_vector[index+3], eigenvector[point]);
            }
          }
        }
      }

      // write spinor field dirac 0
      sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac0, NULL, 1, 64);
      destruct_writer(writer);

      // write spinor field dirac 1
      sprintf(filename, "%s.%04d.%02d.%02d", "source1", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac1, NULL, 1, 64);
      destruct_writer(writer);

      // write spinor field dirac 2
      sprintf(filename, "%s.%04d.%02d.%02d", "source2", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac2, NULL, 1, 64);
      destruct_writer(writer);

      // write spinor field dirac 3
      sprintf(filename, "%s.%04d.%02d.%02d", "source3", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac3, NULL, 1, 64);
      destruct_writer(writer);
    }
  }
  free(eigenvector);
  free(dirac0);
  free(dirac1);
  free(dirac2);
  free(dirac3);
  free(rnd_vector);
  return;
}

void create_source_tb2i16_df_li(const int nr_conf, const int nr_dilution,
    char* inverterpath) {
  char filename[200];
  int tslice = 0, tb = 0, ti = 0, t = 0, vec = 0, v = 0;
  int count = 0, status = 0, point = 0, index = 0;
  int rnd_vec_size = T * 4 * no_eigenvalues;
  su3_vector * eigenvector = NULL;
  spinor *tmp = NULL;
  spinor *dirac0 = NULL;
  spinor *dirac1 = NULL;
  spinor *dirac2 = NULL;
  spinor *dirac3 = NULL;
  WRITER* writer = NULL;
  _Complex double *rnd_vector = NULL;
  FILE* file;

// allocate random vector
  rnd_vector = (_Complex double*) calloc(rnd_vec_size, sizeof(_Complex double));
  if (rnd_vector == NULL ) {
    fprintf(stderr, "Could not allocate random vector!\nAborting...\n");
    exit(-1);
  }
  rnd_z2_vector(rnd_vector, rnd_vec_size);
// filename for the random vector file
  sprintf(filename, "randomvector.%03d.%s.Tib.%04d", nr_dilution,
      dilution_list[nr_dilution].quarktype, nr_conf);
#if DEBUG
  printf("\nwriting random vector to file %s\n", filename);
#endif
// write random vector to file
  if ((file = fopen(filename, "wb")) == NULL ) {
    fprintf(stderr, "Could not open file %s for random vector\nAborting...\n",
        filename);
    exit(-1);
  }
  count = fwrite(rnd_vector, sizeof(_Complex double), rnd_vec_size, file);
// check the number of elements written is correct, otherwise fail!
  if (count != rnd_vec_size) {
    fprintf(stderr,
        "Could not print all data to file %s for random vector (%d of %d)\nAborting...\n",
        filename, count, rnd_vec_size);
    exit(-1);
  }
  fclose(file);
//  allocate spinors and eigenvectors
  eigenvector = (su3_vector*) calloc(SPACEVOLUME, sizeof(su3_vector));
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

// main calculation loop
// here, block-interlace dilution with length 2-16 is hard coded in time direction
// a sanity check for this is already done in the add_dilution routine
  for (tslice = 0; tslice < dilution_list[nr_dilution].size[0]; tslice++) {
    for (vec = 0; vec < dilution_list[nr_dilution].size[2]; vec++) {
      // zero the spinor fields
      zero_spinor_field(dirac0, VOLUMEPLUSRAND);
      zero_spinor_field(dirac1, VOLUMEPLUSRAND);
      zero_spinor_field(dirac2, VOLUMEPLUSRAND);
      zero_spinor_field(dirac3, VOLUMEPLUSRAND);

      // interlace loop in eigenvector space
      for (v = vec; v < no_eigenvalues; v +=
          dilution_list[nr_dilution].size[2]) {
        for (ti = tslice * 2; ti < T; ti += 16) {
          for (tb = -1; tb < 1; tb++) { // negative to get the times T-1 and 0 together!
            t = (ti + tb + T) % T;

            // read in eigenvector and distribute it to the sources
#if BINARYINPUT // binary eigenvector files
            sprintf(filename, "%seigenvectors.%04d.%03d", EIGENSYSTEMPATH,
                nr_conf, t);
#if DEBUG
            printf("reading file %s\n", filename);
#endif
            read_binary_eigenvector(eigenvector, filename, v);
#else  // ILDG format eigenvector files
            sprintf(filename, "%seigenvector.%03d.%03d.%04d", EIGENSYSTEMPATH,
                v, t, nr_conf);
#if DEBUG
            printf("reading file %s\n", filename);
#endif
            read_su3_vector(eigenvector, filename, 0, t, 1);
#endif

            // loop over position space
            index = t * no_eigenvalues * 4 + v * 4;
            for (point = 0; point < SPACEVOLUME; point++) {
              _vector_add_mul( dirac0[SPACEVOLUME*t + point].s0,
                  rnd_vector[index+0], eigenvector[point]);
              _vector_add_mul( dirac1[SPACEVOLUME*t + point].s1,
                  rnd_vector[index+1], eigenvector[point]);
              _vector_add_mul( dirac2[SPACEVOLUME*t + point].s2,
                  rnd_vector[index+2], eigenvector[point]);
              _vector_add_mul( dirac3[SPACEVOLUME*t + point].s3,
                  rnd_vector[index+3], eigenvector[point]);
            }
          }
        }
      }

      // write spinor field dirac 0
      sprintf(filename, "%s.%04d.%02d.%02d", "source0", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac0, NULL, 1, 64);
      destruct_writer(writer);

      // write spinor field dirac 1
      sprintf(filename, "%s.%04d.%02d.%02d", "source1", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac1, NULL, 1, 64);
      destruct_writer(writer);

      // write spinor field dirac 2
      sprintf(filename, "%s.%04d.%02d.%02d", "source2", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac2, NULL, 1, 64);
      destruct_writer(writer);

      // write spinor field dirac 3
      sprintf(filename, "%s.%04d.%02d.%02d", "source3", nr_conf, tslice, vec);
#if DEBUG
      printf("writing file %s\n", filename);
#endif
      construct_writer(&writer, filename, 0);
      status = write_spinor(writer, &dirac3, NULL, 1, 64);
      destruct_writer(writer);
    }
  }
  free(eigenvector);
  free(dirac0);
  free(dirac1);
  free(dirac2);
  free(dirac3);
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
  dptr->seed = d_seed;

// time dilution
  if (dptr->type[0] == D_INTER || dptr->type[0] == D_BLOCK) {
    if (d_t > T || d_t <= 0) {
      fprintf(stderr, "Dilution number in time is <=0 or >T\n Aborting...\n");
      exit(-1);
    } else {
      if (T % d_t != 0) {
        fprintf(stderr,
            "Dilution size is no divisor of time size\n Aborting...\n");
        exit(-1);
      }
      dptr->size[0] = d_t;
    }
    sprintf(dptr->typestring, "%s", (dptr->type[0] == D_INTER) ? "i" : "b");
  } else if (dptr->type[0] == D_FULL) {
    dptr->size[0] = T;
    sprintf(dptr->typestring, "f");
  } else if (dptr->type[0] == D_NONE) {
    dptr->size[0] = 1;
    sprintf(dptr->typestring, "n");
  } else if (dptr->type[0] == D_INTERBLOCK) {
    if (d_t == 1) {
      if (T % 4 != 0) {
        fprintf(stderr,
            "Dilution size is no divisor of time size\n Aborting...\n");
        exit(-1);
      } else {
        dptr->size[0] = T / 4;
        sprintf(dptr->typestring, "ib");
      }
    } else if (d_t == 2) {
      if (T % 6 != 0) {
        fprintf(stderr,
            "Dilution size is no divisor of time size\n Aborting...\n");
        exit(-1);
      } else {
        dptr->size[0] = T / 6;
        sprintf(dptr->typestring, "ib");
      }
    } else if (d_t == 3) {
      if (T % (2 * T / 16) != 0) {
        fprintf(stderr,
            "Dilution size is no divisor of time size\n Aborting...\n");
        exit(-1);
      } else {
        dptr->size[0] = T / (2 * T / 16);
        sprintf(dptr->typestring, "ib");
      }
    } else {
      fprintf(stderr,
          "Inter-block dilution scheme not recognized!\n Aborting...\n");
      exit(-1);
    }
  } else {
    fprintf(stderr, "Dilution scheme for time not recognized!\nAborting...\n");
    exit(-2);
  }

// spin dilution
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

// LapH dilution
  if (dptr->type[2] == D_INTER || dptr->type[2] == D_BLOCK) {
    if (d_l > no_eigenvalues || d_l <= 0) {
      fprintf(stderr,
          "Dilution number in LapH space is <=0 or >#eigenvalues\n Aborting...\n");
      exit(-1);
    } else {
      if (no_eigenvalues % d_l != 0) {
        fprintf(stderr,
            "Dilution size is no divisor of LapH size\n Aborting...\n");
        exit(-1);
      }
      dptr->size[2] = d_l;
    }
  } else if (dptr->type[2] == D_FULL) {
    dptr->size[2] = no_eigenvalues;
  } else if (dptr->type[2] == D_NONE) {
    dptr->size[2] = 1;
  } else {
    fprintf(stderr, "Dilution scheme for LapH not recognized!\nAborting...\n");
    exit(-2);
  }

// quark type
  if (quark_type == D_UP) {
    sprintf(dptr->quarktype, "u");
  } else if (quark_type == D_DOWN) {
    sprintf(dptr->quarktype, "d");
  } else if (quark_type == D_STRANGE) {
    sprintf(dptr->quarktype, "s");
  } else if (quark_type == D_CHARM) {
    sprintf(dptr->quarktype, "c");
  } else {
    fprintf(stderr, "Quark type not recognized!\nAborting...\n");
    exit(-2);
  }

// sink
  if (smearing == D_STOCH || smearing == D_LOCAL) {
    dptr->smearing = smearing;
  } else {
    fprintf(stderr, "Smearing type not recognized!\nAborting...\n");
    exit(-2);
  }

//#if DEBUG
//  printf("dilution %d:\n", no_dilution);
//  printf("time: %s (%d), %d\n", dptr->typestring, dptr->type[0], dptr->size[0]);
//  printf("spin: %d, %d\n", dptr->type[1], dptr->size[1]);
//  printf("LapH: %d, %d\n", dptr->type[2], dptr->size[2]);
//  printf("quark: %s\nsmearing: %d\n", dptr->quarktype, dptr->smearing);
//#endif

  no_dilution++;
}
