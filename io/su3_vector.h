/***********************************************************************
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008 Carsten Urbach
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

#ifndef _SU3VECTOR_H
#define _SU3VECTOR_H

#include <su3.h>

#include <io/selector.h>
#include <io/utils.h>

int read_su3_vector(su3_vector * const s, char * filename, const int position, const int t0);
int read_binary_su3_vector_data(su3_vector * const s, READER * reader, DML_Checksum * checksum, const int t0);

int write_su3_vector(WRITER * writer, double const eigenvalue, su3_vector * const s, const int prec, const int t0);
int write_binary_su3_vector_data(su3_vector * const s, WRITER * writer, DML_Checksum *checksum, int const prec, const int t0);

void write_su3_vector_info(WRITER * writer, int append);
//void write_su3_vector_format(WRITER *writer, paramsPropagatorFormat const *format);
//void write_su3_vector_type(WRITER *writer, const int type);

#endif
