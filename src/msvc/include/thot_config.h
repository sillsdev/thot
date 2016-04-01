#ifndef _THOT_CONFIG_H
#define _THOT_CONFIG_H

#pragma warning(disable:4996)

#ifndef THOT_LM_TYPE
#define THOT_LM_TYPE THOT_INCR_JEL_MER_LM
#endif

#ifndef THOT_PBM_TYPE
#define THOT_PBM_TYPE ML_PBM
#endif

#ifndef THOT_SMT_MODEL_TYPE
#define THOT_SMT_MODEL_TYPE PBLSWMLI
#endif

#ifndef THOT_MSTACK_TYPE
#define THOT_MSTACK_TYPE MSTACK
#endif

#ifndef THOT_SWM_TYPE
#define THOT_SWM_TYPE THOT_INCR_HMM_P0_SWM
#endif

#ifndef THOT_AT_TYPE
#define THOT_AT_TYPE WG_UNCTRANS
#endif

#ifndef THOT_ECM_NB_UCAT_TYPE
#define THOT_ECM_NB_UCAT_TYPE NONPB_ECM_NB_UCAT
#endif

#ifndef THOT_ECM_TYPE
#define THOT_ECM_TYPE PFSM_FOR_WG_ECM
#endif

#ifndef THOT_WGP_TYPE
#define THOT_WGP_TYPE STD_WGP
#endif

#define THOT_DISABLE_PREPROC_CODE

#define M_LN10 log(10)

#define ftello ftell
#define fseeko fseek

typedef int ssize_t;

#endif