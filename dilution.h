#ifndef DILUTION_H_
#define DILUTION_H_

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
 *
 * File dilution.h
 *
 * Type definitions and macros for stochastic estimations
 *
 *******************************************************************************/

#define max_no_dilution 35

#define D_NONE 0
#define D_FULL 1
#define D_BLOCK 2
#define D_INTER 3

#define D_UP 0
#define D_DOWN 1

#define D_STOCH 0
#define D_LOCAL 1

typedef struct {
  // type of the dilution: none, full, block or interlace
  int type[3];

  // size in time, spin and eigenvector space
  int size[3];

  // seed for the random number generator
  int seed;

  // quark type
  int quark;

  // smearing
  int smearing;

} dilution;

extern int g_stochastical_run;
extern int no_dilution;
extern dilution dilution_list[max_no_dilution];

void add_dilution(const int d_type_t, const int d_type_d, const int d_type_l,
    const int d_t, const int d_d, const int d_l, const int d_seed,
    const int quark_type, const int smearing);

void create_source_tf_df_lf(const int nr_conf, const int nr_dilution,
    char* inverterpath);
void create_source_tf_df_ln(const int nr_conf, const int nr_dilution,
    char* inverterpath);
void create_source_tf_df_li(const int nr_conf, const int nr_dilution,
    char* inverterpath);
void create_source_tf_df_lb(const int nr_conf, const int nr_dilution,
    char* inverterpath);

void create_source_tf_df_li1(const int nr_conf, const int nr_dilution,
    char* inverterpath, int *tslices, int nr_tslices);

void create_source_ti_df_lf(const int nr_conf, const int nr_dilution,
    char* inverterpath);
void create_source_ti_df_ln(const int nr_conf, const int nr_dilution,
    char* inverterpath);
void create_source_ti_df_li(const int nr_conf, const int nr_dilution,
    char* inverterpath);
void create_source_ti_df_lb(const int nr_conf, const int nr_dilution,
    char* inverterpath);


#endif /* DILUTION_H_ */
