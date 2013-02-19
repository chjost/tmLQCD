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

#ifdef HAVE_LIBLEMON
// this cannot work like this...
int write_binary_ranlux_data(int * const s,
		LemonWriter * lemonwriter, DML_Checksum *checksum) {
//	int x, y, z, t, i = 0, xG, yG, zG, tG, status = 0;
//	int latticeSize[] = {1, g_nproc_x*LX, g_nproc_y*LY, g_nproc_z*LZ};
//	int scidacMapping[] = {0, 3, 2, 1};
//	unsigned long bufoffset = 0;
//	char *filebuffer = NULL;
//	uint64_t bytes;
//	DML_SiteRank rank;
//	double tick = 0, tock = 0;
//	char measure[64];
//
//	DML_checksum_init(checksum);
//	bytes = (uint64_t)sizeof(su3_vector);
//	if (prec == 32) {
//		bytes /= 2;
//	}
//	if((void*)(filebuffer = malloc(VOLUME * bytes)) == NULL) {
//		fprintf (stderr, "malloc errno in write_binary_su3_vector_data_parallel: %d\n", errno);
//		fflush(stderr);
//		errno = 0;
//		/* do we need to abort here? */
//		return 1;
//	}
//
//	tG = g_proc_coords[0]*T;
//	zG = g_proc_coords[3]*LZ;
//	yG = g_proc_coords[2]*LY;
//	xG = g_proc_coords[1]*LX;
//	for(z = 0; z < LZ; z++) {
//		for(y = 0; y < LY; y++) {
//			for(x = 0; x < LX; x++) {
//				rank = (DML_SiteRank) (((zG + z)*L + yG + y)*L + xG + x);
//				i = g_ipt[t][x][y][z];
//
//				if (prec == 32)
//					be_to_cpu_assign_double2single((float*)(filebuffer + bufoffset), (double*)(s + i), sizeof(su3_vector) / 8);
//				else
//					be_to_cpu_assign((double*)(filebuffer + bufoffset), (double*)(s + i),  sizeof(su3_vector) / 8);
//				DML_checksum_accum(checksum, rank, (char*) filebuffer + bufoffset, bytes);
//				bufoffset += bytes;
//			}
//		}
//	}
//}
//
//if (g_debug_level > 0) {
//	MPI_Barrier(g_cart_grid);
//	tick = MPI_Wtime();
//}
//
//status = lemonWriteLatticeParallelMapped(lemonwriter, filebuffer, bytes, latticeSize, scidacMapping);
//
//if (status != LEMON_SUCCESS)
//{
//	free(filebuffer);
//	fprintf(stderr, "LEMON write error occurred with status = %d, while in write_binary_su3_vector_data_l (su3_vector_write_binary.c)!\n", status);
//	return(-2);
//}
//
//if (g_debug_level > 0) {
//	MPI_Barrier(g_cart_grid);
//	tock = MPI_Wtime();
//
//	if (g_cart_id == 0) {
//		engineering(measure, latticeSize[0] * latticeSize[1] * latticeSize[2] * latticeSize[3] * bytes, "b");
//		fprintf(stdout, "# Time spent writing %s ", measure);
//		engineering(measure, tock - tick, "s");
//		fprintf(stdout, "was %s.\n", measure);
//		engineering(measure, latticeSize[0] * latticeSize[1] * latticeSize[2] * latticeSize[3] * bytes / (tock - tick), "b/s");
//		fprintf(stdout, "# Writing speed: %s", measure);
//		engineering(measure, latticeSize[0] * latticeSize[1] * latticeSize[2] * latticeSize[3] * bytes / (g_nproc * (tock - tick)), "b/s");
//		fprintf(stdout, " (%s per MPI process).\n", measure);
//		fflush(stdout);
//	}
//}
//
//lemonWriterCloseRecord(lemonwriter);
//
//DML_global_xor(&checksum->suma);
//DML_global_xor(&checksum->sumb);
//
//free(filebuffer);
	return 0;

}

#else /* HAVE_LIBLEMON */
int write_binary_ranlux_data(int * const state, LimeWriter * limewriter,
		DML_Checksum * checksum, int const length_total) {
	int i = 0, tmp, status = 0;
	n_uint64_t bytes = sizeof(int);
	if (g_cart_id == 0) {
		for (i = 0; i < length_total; i++) {
			tmp = state[i];
			DML_checksum_accum(checksum, (uint32_t) i, (char*) &tmp,
					sizeof(int));
			status = limeWriteRecordData((int *) &tmp, &bytes, limewriter);
			if (status < 0) {
				fprintf(stderr,
						"LIME write error occurred with status = %d, while in write_binary_ranlux_data (ranlux_write_binary.c)!\n",
						status);
#ifdef MPI
				MPI_Abort(MPI_COMM_WORLD, 1);
				MPI_Finalize();
#endif
				exit(500);
			}
		}
	}
	return (0);
}
#endif /* HAVE_LIBLEMON */
