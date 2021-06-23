/*
thot package for statistical machine translation
Copyright (C) 2013 Daniel Ortiz-Mart\'inez

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program; If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * @file getdelim.h
 * @brief Defines the getdelim function
 */

#pragma once

#define _WITH_GETLINE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#if THOT_HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifndef THOT_HAVE_SSIZE_T
typedef int ssize_t;
#endif

#ifdef __cplusplus
extern "C"
{
#endif
#ifndef THOT_HAVE_GETDELIM
  ssize_t getdelim(char** lineptr, size_t* n, int delimiter, FILE* stream);
#endif
#ifdef __cplusplus
}
#endif

