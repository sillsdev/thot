#ifndef _THOT_H_
#define _THOT_H_

#if defined _WIN32 || defined __CYGWIN__
  #if defined THOT_EXPORTING
    #if defined __GNUC__
      #define THOT_API __attribute__((dllexport))
    #else
      #define THOT_API __declspec(dllexport)
    #endif
  #else
    #if defined __GNUC__
      #define THOT_API __attribute__((dllimport))
    #else
      #define THOT_API __declspec(dllimport)
    #endif
  #endif
#elif __GNUC__ >= 4
  #define THOT_API __attribute__ ((visibility("default")))
#else
  #define THOT_API
#endif

#ifdef __cplusplus
extern "C"
{
#endif

THOT_API void* decoder_open(const char* cfgFileName);

THOT_API void* decoder_openSession(void* decoderHandle);

THOT_API void decoder_saveModels(void* decoderHandle);

THOT_API void* decoder_getSingleWordAlignmentModel(void* decoderHandle);

THOT_API void* decoder_getInverseSingleWordAlignmentModel(void* decoderHandle);

THOT_API void decoder_close(void* decoderHandle);

THOT_API void* session_translate(void* sessionHandle,const char* sentence);

THOT_API unsigned int session_translateNBest(void* sessionHandle,unsigned int n,const char* sentence,void** results);

THOT_API void* session_getBestPhraseAlignment(void* sessionHandle,const char* sentence,const char* translation);

THOT_API void* session_translateInteractively(void* sessionHandle,const char* sentence);

THOT_API void* session_addStringToPrefix(void* sessionHandle,const char* addition);

THOT_API void* session_setPrefix(void* sessionHandle,const char* prefix);

THOT_API void session_trainSentencePair(void* decoderHandle,const char* sourceSentence,const char* targetSentence,int** matrix,unsigned int iLen,unsigned int jLen);

THOT_API void session_close(void* sessionHandle);

THOT_API unsigned int tdata_getTarget(void* dataHandle,char* target,unsigned int capacity);

THOT_API unsigned int tdata_getPhraseCount(void* dataHandle);

THOT_API unsigned int tdata_getSourceSegmentation(void* dataHandle,unsigned int** sourceSegmentation,unsigned int capacity);

THOT_API unsigned int tdata_getTargetSegmentCuts(void* dataHandle,unsigned int* targetSegmentCuts,unsigned int capacity);

THOT_API unsigned int tdata_getTargetUnknownWords(void* dataHandle,unsigned int* targetUnknownWords,unsigned int capacity);

THOT_API double tdata_getScore(void* dataHandle);

THOT_API unsigned int tdata_getScoreComponents(void* dataHandle,double* scoreComps,unsigned int capacity);

THOT_API void tdata_destroy(void* dataHandle);

THOT_API void* swAlignModel_create();

THOT_API void* swAlignModel_open(const char* prefFileName);

THOT_API void swAlignModel_addSentencePair(void* swAlignModelHandle,const char* sourceSentence,const char* targetSentence,int** matrix,unsigned int iLen,unsigned int jLen);

THOT_API void swAlignModel_train(void* swAlignModelHandle,unsigned int numIters);

THOT_API void swAlignModel_save(void* swAlignModelHandle,const char* prefFileName);

THOT_API float swAlignModel_getTranslationProbability(void* swAlignModelHandle,const char* srcWord,const char* trgWord);

THOT_API float swAlignModel_getBestAlignment(void* swAlignModelHandle,const char* sourceSentence,const char* targetSentence,int** matrix,unsigned int* iLen,unsigned int* jLen);

THOT_API void swAlignModel_close(void* swAlignModelHandle);

THOT_API bool giza_symmetr1(const char* lhsFileName,const char* rhsFileName,const char* outputFileName,bool transpose);

THOT_API bool phraseModel_generate(const char* alignmentFileName,int maxPhraseLength,const char* tableFileName);

THOT_API void* langModel_open(const char* prefFileName);

THOT_API float langModel_getSentenceProbability(void* lmHandle,const char* sentence);

THOT_API void langModel_close(void* lmHandle);

#ifdef __cplusplus
}
#endif

#endif