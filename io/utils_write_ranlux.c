/***********************************************************************
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

#include "utils.ih"

void write_ranlux_xml(WRITER * writer) {
	char *message;
	uint64_t bytes;

	message = (char*) malloc(512);
	if (message == (char*) NULL)
		kill_with_error(writer->fp, g_cart_id,
				"Memory allocation error in write_ranlux. Aborting\n");

	sprintf(message, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
			"<ranlux-info>\n"
			"  <seed>%i</seed>\n"
			"  <rxld_level>%i</rxld_level>\n"
			"</ranlux-info>", random_seed, rlxd_level);
	bytes = strlen(message);

	write_header(writer, 1, 0, "ranlux-info", bytes);
	write_message(writer, message, bytes);

	free(message);
	return;
}
