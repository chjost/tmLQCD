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
#include "default_input_values.h"

int read_su3_vector(su3_vector * const s, char * filename, const int position_,
    const int t0, const int nsets) {
  int status = 0, read_data = 0, getpos = 0, bytes = 0, prec = 0, prop_type,
      position = position_, rstat = 0;
  int read_checksum = 0, status_checksum = 0, position_checksum = position,
      getpos_checksum = 0;
  int read_eigenvalue = 0, status_eigenvalue = 0, position_eigenvalue = 0,
      getpos_eigenvalue = 0;
  int sets = 0, timeslice = 0;
  char *header_type = NULL;
  READER *reader = NULL;
  DML_Checksum checksum, checksum_read;
  char *buffer = NULL;
  double eigenvalue = 0.0;
  construct_reader(&reader, filename);

  position_checksum = position;
  position_eigenvalue = position;

  DML_checksum_init(&checksum_read);

  /* Find the desired propagator (could be more than one in a file) */
  while ((status = ReaderNextRecord(reader)) != LIME_EOF) {
    if (status != LIME_SUCCESS) {
      fprintf(stderr, "ReaderNextRecord returned status %d.\n", status);
      break;
    }
    header_type = ReaderType(reader);
/*
 * read the eigenvalue-info package with the number of eigenvectors saved,
 * the first eigenvalue and the timeslice the vectors belong to
 */
    if (strcmp("eigenvalue-info", header_type) == 0 && read_eigenvalue == 0) {
      if (position_eigenvalue == getpos_eigenvalue) {
        status_eigenvalue = read_message(reader, &buffer);
        if (status_eigenvalue != LIME_SUCCESS) {
          fprintf(stderr, "eigenvalue reading failed with return value %d\n",
              status_eigenvalue);
        } else {
          status_eigenvalue = parse_eigenvalue_xml(buffer, &eigenvalue,
              &timeslice, &sets);
          if (status_eigenvalue == 0) {
            fprintf(stderr,
                "extracting eigenvalue failed with return value %d\n",
                status_eigenvalue);
          } else {
            if (t0 != timeslice) {
              fprintf(stderr,
                  "eigenvector of the wrong timeslice!\nAborting...\n");
              return (-9);
            }
            if (nsets != sets) {
              fprintf(stderr,
                  "wrong number of eigenvectors!\nAborting...\n");
              return (-10);
            }
          }
        }
        read_eigenvalue++;
      }
      getpos_eigenvalue++;
    }
/*
 * read the actual data for the eigenvectors
 */
    if (strcmp("scidac-binary-data", header_type) == 0 && read_data == 0) {
      if (getpos == position) {
        bytes = ReaderBytes(reader);

        if ((int) bytes
            == LX * g_nproc_x * LY * g_nproc_y * LZ * g_nproc_z * sets
                * sizeof(su3_vector)) {
          prec = 64;
        } else {
          if ((int) bytes
              == LX * g_nproc_x * LY * g_nproc_y * LZ * g_nproc_z * sets
                  * sizeof(su3_vector) / 2) {
            prec = 32;
          } else {
            fprintf(stderr,
                "Length of scidac-binary-data record in %s does not match input parameters.\n",
                filename);
            fprintf(stderr, "Found %d bytes.\n", bytes);
            return (-6);
          }
        }

        if (g_cart_id == 0 && g_debug_level > 0) {
          printf("# %s precision read (%d bits).\n",
              (prec == 64 ? "Double" : "Single"), prec);
        }

        if ((rstat = read_binary_su3_vector_data(s, reader, &checksum, t0, sets))
            != 0) {
          fprintf(stderr,
              "read_binary_su3_vector_data failed with return value %d\n",
              rstat);
          return (-7);
        }

        read_data++;
      }
      ++getpos;
    }
/*
 * read the checksum
 */
    if (strcmp("scidac-checksum", header_type) == 0 && read_checksum == 0) {
      if (position_checksum == getpos_checksum) {
        status_checksum = read_message(reader, &buffer);
        if (status_checksum != LIME_SUCCESS) {
          fprintf(stderr, "checksum reading failed with return value %d\n",
              status_checksum);
        } else {
          status_checksum = parse_checksum_xml(buffer, &checksum_read);
          if (status_checksum == 0) {
            fprintf(stderr, "extracting checksum failed with return value %d\n",
                status_checksum);
          }
        }
        read_checksum++;
      }
      getpos_checksum++;
    }

    if (read_data && read_checksum && read_eigenvalue)
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

  if (g_cart_id == 0 && g_debug_level >= 3) {
    printf("# Scidac checksums for eigenvector stored in %s position %d:\n",
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
