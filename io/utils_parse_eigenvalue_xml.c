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

#include "utils.ih"

int parse_eigenvalue_xml(char *message, double *eigenvalue, int *timeslice)
{
  int  read_value = 0, read_timeslice = 0;
  char *pos = strtok(message, "<> \n\t");

  while (pos)
  {
    if (!strncmp(pos, "value", 4)) {
      pos = strtok(0, "<> \n\t");
      sscanf(pos, "%f", eigenvalue);
      read_value = 1;
    }
    if (!strncmp(pos, "timeslice", 4)) {
      pos = strtok(0, "<> \n\t");
      sscanf(pos, "%d", timeslice);
      read_timeslice = 1;
    }
    pos = strtok(0, "<> \n\t");
  }
  return (read_value && read_timeslice);
}
