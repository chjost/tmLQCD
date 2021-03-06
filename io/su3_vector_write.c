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


#include "su3_vector.ih"

int write_su3_vector(WRITER * writer, double const *eigenvalue, su3_vector * const s, const int prec, const int t0, int const nsets)
{
	DML_Checksum checksum;
	uint64_t bytes;
	int status = 0;
//	general information about the configuration
	write_su3_vector_info(writer, 0);

//	eigenvalue
	write_eigenvalue_xml(writer, eigenvalue, t0, nsets);

// data
	bytes = (n_uint64_t)LX * g_nproc_x * LY * g_nproc_y * LZ * g_nproc_z * (n_uint64_t)(sizeof(su3_vector) * prec / 64) * nsets;

	write_header(writer, 0, 0, "scidac-binary-data", bytes);
	status  = write_binary_su3_vector_data(s, writer, &checksum, prec, t0, nsets);
	write_checksum(writer, &checksum, NULL);

	return status;
}
