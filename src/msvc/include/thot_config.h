#ifndef _THOT_CONFIG_H
#define _THOT_CONFIG_H

#pragma warning(disable:4996)

#define THOT_LM_TYPE THOT_INCR_JEL_MER_LM
#define THOT_PBM_TYPE  ML_PBM
#define THOT_SMT_MODEL_TYPE  PBLSWMLI
#define THOT_MSTACK_TYPE  MSTACK
#define THOT_SWM_TYPE  THOT_SINCR_IBM2_SWM
#define THOT_AT_TYPE  WG_UNCTRANS
#define THOT_ECM_NB_UCAT_TYPE  NONPB_ECM_NB_UCAT
#define THOT_ECM_TYPE  PFSM_FOR_WG_ECM
#define THOT_WGP_TYPE  STD_WGP

#define THOT_DISABLE_PREPROC_CODE

#define M_LN10 log(10)

#define ftello ftell
#define fseeko fseek

typedef int ssize_t;

#endif