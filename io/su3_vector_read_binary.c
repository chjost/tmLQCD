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

#ifdef HAVE_LIBLEMON
int read_binary_su3_vector_data(su3_vector * const s, READER * READER, DML_Checksum *checksum, const int t0) {

//	int t, x, y , z, i = 0, status = 0;
//	int latticeSize[] = {T_global, g_nproc_x*LX, g_nproc_y*LY, g_nproc_z*LZ};
//	int scidacMapping[] = {0, 3, 2, 1};
//	int prec = 0;
//	n_uint64_t bytes;
//	char *filebuffer = NULL, *current = NULL;
//	double tick = 0, tock = 0;
//	DML_SiteRank rank;
//	char measure[64];
//
//	bytes = ReaderBytes(READER);
//
//	if (bytes == (n_uint64_t)g_nproc * (n_uint64_t)VOLUME * (n_uint64_t)sizeof(su3_vector)) {
//		prec = 64;
//		bytes = sizeof(su3_vector);
//	}
//	else {
//		if (bytes == (n_uint64_t)g_nproc * (n_uint64_t)VOLUME * (n_uint64_t)sizeof(su3_vector) / 2) {
//			prec = 32;
//			bytes = sizeof(su3_vector)/2;
//		}
//		else {
//			return(-3);
//		}
//	}
//
//	DML_checksum_init(checksum);
//
//	if((void*)(filebuffer = malloc(VOLUME * bytes)) == NULL) {
//		return(-1);
//	}
//
//	if (g_debug_level > 0) {
//		MPI_Barrier(g_cart_grid);
//		tick = MPI_Wtime();
//	}
//	status = lemonReadLatticeParallelMapped(READER, filebuffer, bytes, latticeSize, scidacMapping);
//
//	if (g_debug_level > 0) {
//		MPI_Barrier(g_cart_grid);
//		tock = MPI_Wtime();
//
//		if (g_cart_id == 0) {
//			engineering(measure, latticeSize[0] * latticeSize[1] * latticeSize[2] * latticeSize[3] * bytes, "b");
//			fprintf(stdout, "# Time spent reading %s ", measure);
//			engineering(measure, tock - tick, "s");
//			fprintf(stdout, "was %s.\n", measure);
//			engineering(measure, latticeSize[0] * latticeSize[1] * latticeSize[2] * latticeSize[3] * bytes / (tock - tick), "b/s");
//			fprintf(stdout, "# Reading speed: %s", measure);
//			engineering(measure, latticeSize[0] * latticeSize[1] * latticeSize[2] * latticeSize[3] * bytes / (g_nproc * (tock - tick)), "b/s");
//			fprintf(stdout, " (%s per MPI process).\n", measure);
//			fflush(stdout);
//		}
//	}
//
//	if (status < 0 && status != LEMON_EOR) {
//		fprintf(stderr, "lemonReadLatticeParallelMapped returned error %d in su3_vector_read_binary.c", status);
//		free(filebuffer);
//		return(-2);
//	}
//
//	for (t = 0; t < T; t++) {
//		for (z = 0; z < LZ; z++) {
//			for (y = 0; y < LY; y++) {
//				for (x = 0; x < LX; x++) {
//					rank = (DML_SiteRank)(g_proc_coords[1] * LX +
//							(((g_proc_coords[0] * T + t) * g_nproc_z * LZ
//											+ g_proc_coords[3] * LZ + z) * g_nproc_y * LY
//									+ g_proc_coords[2] * LY + y) *
//							((DML_SiteRank) LX * g_nproc_x) + x);
//					current = filebuffer + bytes * (x + (y + (t * LZ + z) * LY) * LX);
//					DML_checksum_accum(checksum, rank, current, bytes);
//
//					i = g_ipt[t][x][y][z];
//					if (prec == 32)
//					be_to_cpu_assign_single2double(s + i, current, sizeof(su3_vector) / 8);
//					else
//					be_to_cpu_assign(s + i, current, sizeof(su3_vector) / 8);
//				}
//			}
//		}
//	}
//
//	DML_global_xor(&checksum->suma);
//	DML_global_xor(&checksum->sumb);
//
//	free(filebuffer);
  return 0;
}
#else /* HAVE_LIBLEMON */
int read_binary_su3_vector_data(su3_vector * const s, READER * reader,
    DML_Checksum * checksum, const int t0, int const nsets) {
  int t, x, y, z, i = 0, status = 0;
  int X, Y, Z;
  n_uint64_t bytes;
  su3_vector tmp[1];
  float tmp2[24];
  DML_SiteRank rank;
  int prec;
  int sets = 0;

  DML_checksum_init(checksum);
  bytes = ReaderBytes(reader);

  if (bytes
      == (n_uint64_t) g_nproc * (n_uint64_t) (SPACEVOLUME + SPACERAND)
          * (n_uint64_t) sizeof(su3_vector) * nsets) {
    prec = 64;
    bytes = sizeof(su3_vector);
  } else {
    if (bytes
        == (n_uint64_t) g_nproc * (n_uint64_t) (SPACEVOLUME + SPACERAND)
            * (n_uint64_t) sizeof(su3_vector) * nsets / 2) {
      prec = 32;
      bytes = sizeof(su3_vector) / 2;
    } else {
      return (-3);
    }
  }
  t =  T * g_proc_coords[0];
//  t = t0 - T * g_proc_coords[0];
  for (sets = 0; sets < nsets; sets++) {

    for (z = 0; z < LZ; z++) {
      Z = z - g_proc_coords[3] * LZ;
      for (y = 0; y < LY; y++) {
        Y = y - g_proc_coords[2] * LY;
#if (defined MPI)
        readerSeek(reader,(n_uint64_t) (g_proc_coords[1]*LX +
                (((g_proc_coords[0]*T+t)*g_nproc_z*LZ + g_proc_coords[3]*LZ+z)*g_nproc_y*LY
                    + g_proc_coords[2]*LY+y)*LX*g_nproc_x)*bytes,
            SEEK_SET);
#endif
        for (x = 0; x < LX; x++) {
          X = x - g_proc_coords[1] * LX;
          i = g_ipt[t][X][Y][Z];
          rank = (DML_SiteRank) ((z * LY * g_nproc_y + y) * LX * g_nproc_x + x);
          if (prec == 32) {
            status = ReaderReadData(tmp2, &bytes, reader);
            DML_checksum_accum(checksum, rank, (char *) tmp2, bytes);
            be_to_cpu_assign_single2double(s + i, (float*) tmp2,
                sizeof(su3_vector) / 8);
          } else {
            status = ReaderReadData(tmp, &bytes, reader);
            DML_checksum_accum(checksum, rank, (char *) tmp, bytes);
            be_to_cpu_assign(s + i, tmp, sizeof(su3_vector) / 8);
          }
          if (status < 0 && status != LIME_EOR) {
            fprintf(stderr,
                "LIME read error occurred with status = %d while reading in su3_vector_read_binary.c!\n",
                status);
#ifdef MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Finalize();
#endif
            return (-2);
          }
        }
      }
    }
  }

#ifdef MPI
  DML_checksum_combine(checksum);
#endif
  return (0);
}
#endif /* HAVE_LIBLEMON */
