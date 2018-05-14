#ifndef _THOT_CONFIG_H
#define _THOT_CONFIG_H

#pragma warning(disable:4996)

#ifndef THOT_DISABLE_DYNAMIC_LOADING
#define THOT_DISABLE_DYNAMIC_LOADING
#endif

#ifndef THOT_LM_STATE_H
#define THOT_LM_STATE_H "LM_State.h"
#endif

#ifndef THOT_PPINFO_H
#define THOT_PPINFO_H "PpInfo.h"
#endif

#ifndef THOT_SMTMODEL_H
#define THOT_SMTMODEL_H "SmtModel.h"
#endif

#ifndef THOT_DISABLE_PREPROC_CODE
#define THOT_DISABLE_PREPROC_CODE 1
#endif

#ifndef THOT_DISABLE_PHRASE_COUNT_CACHING
#define THOT_DISABLE_PHRASE_COUNT_CACHING
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
#define THOT_VERSION  "3.1.0Beta" 
#endif

#endif