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

#include "nlp_common/ctimer.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64

// MSVC defines this in winsock2.h!?
typedef struct timeval
{
  long tv_sec;
  long tv_usec;
} timeval;

struct timezone
{
  int tz_minuteswest; /* minutes W of Greenwich */
  int tz_dsttime;     /* type of dst correction */
};

int gettimeofday(struct timeval* tp, struct timezone* tzp)
{
  // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
  static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

  SYSTEMTIME system_time;
  FILETIME file_time;
  uint64_t time;

  GetSystemTime(&system_time);
  SystemTimeToFileTime(&system_time, &file_time);
  time = ((uint64_t)file_time.dwLowDateTime);
  time += ((uint64_t)file_time.dwHighDateTime) << 32;

  tp->tv_sec = (long)((time - EPOCH) / 10000000L);
  tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
  return 0;
}
#endif

static double firstcall;

/**
 *
 * @brief Function for time measuring
 *
 * @param elapsed elapsed time since the first call to ctimer (miliseconds.)
 * @param ucpu    cpu elapsed time since the first call to ctimer (miliseconds.)
 * @param scpu    system elapsed time since the first call to ctimer (miliseconds.)
 * @return 0 if THOT_OK.
 *
 * The function must be executed at least two times.
 * The first one starts the timer and sets elapsed to zero.
 * The second execution of ctimer returns the elapsed time between
 * the first and the second execution.
 * The following calls to ctimer return the elapsed time between the first
 * call and the current call.
 *
 */
void ctimer(double* elapsed, double* ucpu, double* scpu)
{
#ifdef THOT_MINGW
  fprintf(stderr, "Warning: ctimer function not implemented for mingw host!");
#else
  struct timeval tm;
  struct timezone tz;
#ifndef _WIN32
  struct tms sistema;
#endif
  double usegs;

  gettimeofday(&tm, &tz);
#ifndef _WIN32
  times(&sistema);
#endif

  usegs = tm.tv_usec + tm.tv_sec * 1E6;

  if (firstcall)
  {
    *elapsed = usegs - firstcall;
  }
  else
  {
    *elapsed = 0;
    firstcall = usegs;
  }

  *elapsed = *elapsed / 1E6;
#ifndef _WIN32
  *ucpu = (double)sistema.tms_utime / (double)CLOCKS_PER_SEC * 1E4;
  *scpu = (double)sistema.tms_stime / (double)CLOCKS_PER_SEC * 1E4;
#endif
#endif
}
