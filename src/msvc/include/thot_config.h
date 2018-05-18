#ifndef _THOT_CONFIG_H
#define _THOT_CONFIG_H

#pragma warning(disable:4996)

#ifndef THOT_LM_STATE_H
#define THOT_LM_STATE_H "LM_State.h"
#endif

#ifndef THOT_PPINFO_H
#define THOT_PPINFO_H "PpInfo.h"
#endif

#ifndef THOT_SMTMODEL_H
#define THOT_SMTMODEL_H "SmtModelLegacy.h"
#endif

#ifndef THOT_DISABLE_PREPROC_CODE
#define THOT_DISABLE_PREPROC_CODE 1
#endif

#ifndef THOT_LIBDIR
#define THOT_LIBDIR "."
#endif

#ifndef THOT_ALT_LIBDIR
#define THOT_ALT_LIBDIR "."
#endif

#ifndef THOT_LIBDIR_VARNAME
#define THOT_LIBDIR_VARNAME "THOT_LIBDIR"
#endif

#ifndef M_LN10
#define M_LN10 log(10)
#endif

#ifndef ftello
#define ftello ftell
#endif

#ifndef fseeko
#define fseeko fseek
#endif

#ifndef HAVE_STRUCT_TIMESPEC
#define HAVE_STRUCT_TIMESPEC
#endif

#ifndef _SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS
#define _SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS
#endif

#ifndef _WIN32_WINNT
#define _WIN32_WINNT _WIN32_WINNT_VISTA
#endif

typedef int ssize_t;

#ifndef THOT_VERSION 
#define THOT_VERSION  "3.2.0Beta" 
#endif

#endif