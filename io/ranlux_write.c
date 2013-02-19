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


#include "ranlux.ih"

int write_ranlux(char * filename, const int append)
{
	DML_Checksum checksum;
	uint64_t bytes;
	int status = 0, size = rlxd_size(), size_total = rlxd_size() + rlxs_size(), *state = NULL;
	WRITER * writer;
	construct_writer(&writer,filename, append);

	write_ranlux_xml(writer);

	state = (int*) calloc(size_total, sizeof(int));
	rlxd_get((&state[0]));
	rlxs_get((&state[size]));
	bytes = (uint64_t) size_total * sizeof(int);

// data
	write_header(writer, 0, 0, "ranlux-data", bytes);
	status  = write_binary_ranlux_data(state, writer, &checksum, size_total);
	write_checksum(writer, &checksum, NULL);

	destruct_writer(writer);
	return status;
}
