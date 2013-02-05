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
#include "default_input_values.h"

int read_ranlux(char * filename, const int position_) {
	int status = 0, read_data = 0, getpos = 0, bytes = 0, prec = 0, prop_type,
			position = position_, rstat = 0;
	int read_checksum = 0, status_checksum = 0, position_checksum = position,
			getpos_checksum = 0;
	int read_ranlux = 0, status_ranlux = 0, position_ranlux = 0, getpos_ranlux =
			0;
	int level = 0, seed = 0;
	char *header_type = NULL;
	READER *reader = NULL;
	DML_Checksum checksum, checksum_read;
	char *buffer = NULL;
	double eigenvalue = 0.0;
	int * state = NULL;
	int length1 = rlxd_size(), length_total = rlxd_size() + rlxs_size();
	state = (int*) calloc(length_total, sizeof(int));
	if(state == (int*) NULL) {
		fprintf(stderr, "Could not initialize ranlux state in read_ranlux.\nAborting...\n");
		exit(500);
	}
	construct_reader(&reader, filename);

	position_checksum = position;
	position_ranlux = position;

	DML_checksum_init(&checksum_read);

	/* Find the desired propagator (could be more than one in a file) */
	while ((status = ReaderNextRecord(reader)) != LIME_EOF) {
		if (status != LIME_SUCCESS) {
			fprintf(stderr, "ReaderNextRecord returned status %d.\n", status);
			break;
		}
		header_type = ReaderType(reader);

		if (strcmp("ranlux-info", header_type) == 0 && read_ranlux == 0) {
			if (position_ranlux == getpos_ranlux) {
				status_ranlux = read_message(reader, &buffer);
				if (status_ranlux != LIME_SUCCESS) {
					fprintf(stderr,
							"ranlux-info reading failed with return value %d\n",
							status_ranlux);
				} else {
					status_ranlux = parse_ranlux_xml(buffer, &seed, &level);
					if (status_ranlux == 0) {
						fprintf(stderr,
								"extracting ranlux-info failed with return value %d\n",
								status_ranlux);
					}
				}
				read_ranlux++;
			}
			getpos_ranlux++;
		}

		if (strcmp("ranlux-data", header_type) == 0 && read_data == 0) {
			if (getpos == position) {
				bytes = ReaderBytes(reader);

				if ((int) bytes != (length_total) * sizeof(int)) {
					fprintf(stderr,
							"Length of ranlux-data record in %s does not match input parameters.\n",
							filename);
					fprintf(stderr, "Found %d bytes.\n", bytes);
					return (-6);
				}
				if ((rstat = read_binary_ranlux_data(state, reader, &checksum, length_total)) != 0) {
					fprintf(stderr,
							"read_binary_ranlux_data failed with return value %d\n",
							rstat);
					return (-7);
				}
				read_data++;
			}
			++getpos;
		}

		if (strcmp("scidac-checksum", header_type) == 0 && read_checksum == 0) {
			if (position_checksum == getpos_checksum) {
				status_checksum = read_message(reader, &buffer);
				if (status_checksum != LIME_SUCCESS) {
					fprintf(stderr,
							"checksum reading failed with return value %d\n",
							status_checksum);
				} else {
					status_checksum = parse_checksum_xml(buffer,
							&checksum_read);
					if (status_checksum == 0) {
						fprintf(stderr,
								"extracting checksum failed with return value %d\n",
								status_checksum);
					}
				}
				read_checksum++;
			}
			getpos_checksum++;
		}

		if (read_data && read_checksum && read_ranlux)
			break;
	}

	if (status == LIME_EOF) {
		fprintf(stderr,
				"Unable to find requested LIME record scidac-binary-data in file %s.\nEnd of file reached before record was found.\n",
				filename);
		return (-5);
	}

	if (read_checksum) {
		if (checksum_read.suma != checksum.suma
				&& checksum_read.sumb != checksum.sumb) {
			fprintf(stderr, "checksums are not equal.\nAborting... \n");
			fflush(stderr);
			fflush(stdout);
			//			return(-8);
		}
	}

	rlxd_reset(&(state[0]));
	rlxs_reset(&(state[length1]));

	if (g_cart_id == 0 && g_debug_level >= 0) {
		printf("# Scidac checksums for ranlux stored in %s position %d:\n",
				filename, position);
		printf("#   Calculated            : A = %#x B = %#x.\n", checksum.suma,
				checksum.sumb);
		if (read_checksum == 1) {
			printf("#   Extracted             : A = %#x B = %#x.\n",
					checksum_read.suma, checksum_read.sumb);
		} else {
			printf(
					"# No Scidac checksum was read from headers, unable to check integrity of file.\n");
		}
	}

	free(buffer);
	destruct_reader(reader);

	return (0);
}
