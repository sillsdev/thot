#ifndef _StandardClasses_h
#define _StandardClasses_h

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include <WordPenaltyModel.h>
#include <IncrJelMerNgramLM.h>
#include <WbaIncrPhraseModel.h>
#include <IncrHmmP0AligModel.h>
#include <PfsmEcmForWg.h>
#include <WgProcessorForAnlp.h>
#include "MiraBleu.h"
#include "KbMiraLlWu.h"
#include "TranslationConstraints.h"
#include "multi_stack_decoder_rec.h"
#include <NonPbEcModelForNbUcat.h>

#define WORD_PENALTY_MODEL WordPenaltyModel
#define NGRAM_LM IncrJelMerNgramLM
#define SW_ALIG_MODEL IncrHmmP0AligModel
#define PHRASE_MODEL WbaIncrPhraseModel
#define EC_MODEL PfsmEcmForWg
#define EC_MODEL_FOR_NB_UCAT NonPbEcModelForNbUcat
#define WG_PROCESSOR_FOR_ANLP WgProcessorForAnlp<PfsmEcmForWg>
#define SCORER MiraBleu
#define LL_WEIGHT_UPDATER KbMiraLlWu
#define TRANS_CONSTRAINTS TranslationConstraints
#define STACK_DECODER multi_stack_decoder_rec<PhrLocalSwLiTm>
#define ASSISTED_TRANSLATOR WgUncoupledAssistedTrans<PhrLocalSwLiTm>

#endif