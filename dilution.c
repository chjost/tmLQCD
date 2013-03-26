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
#include "global.h"
#include "default_input_values.h"
#include "read_input.h"
#include "start.h"
#include "dilution.h"

int g_stochastical_run = 1;
int no_dilution = 0;
dilution dilution_list[max_no_dilution];

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
