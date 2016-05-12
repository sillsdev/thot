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

THOT_API int session_translate(void* sessionHandle, const char* sentence, char* translation, int capacity, void** data);

THOT_API int session_translateInteractively(void* sessionHandle, const char* sentence, char* translation, int capacity, void** data);

THOT_API int session_addStringToPrefix(void* sessionHandle, const char* addition, char* translation, int capacity, void** data);

THOT_API int session_setPrefix(void* sessionHandle, const char* prefix, char* translation, int capacity, void** data);

THOT_API void session_trainSentencePair(void* decoderHandle, const char* sourceSentence, const char* targetSentence);

THOT_API void session_close(void* sessionHandle);

THOT_API int tdata_getPhraseCount(void* dataHandle);

THOT_API int tdata_getSourceSegmentation(void* dataHandle, int** sourceSegmentation, int capacity);

THOT_API int tdata_getTargetSegmentCuts(void* dataHandle, int* targetSegmentCuts, int capacity);

THOT_API int tdata_getUnknownPhrases(void* dataHandle, bool* unknownPhrases, int capacity);

THOT_API void tdata_destroy(void* dataHandle);

THOT_API void* swAlignModel_create();

THOT_API void* swAlignModel_open(const char* prefFileName);

THOT_API void swAlignModel_addSentencePair(void* swAlignModelHandle, const char* sourceSentence, const char* targetSentence);

THOT_API void swAlignModel_train(void* swAlignModelHandle, int numIters);

THOT_API void swAlignModel_save(void* swAlignModelHandle, const char* prefFileName);

THOT_API float swAlignModel_getTranslationProbability(void* swAlignModelHandle, const char* srcWord, const char* trgWord);

THOT_API float swAlignModel_getBestAlignment(void* swAlignModelHandle, const char* sourceSentence, const char* targetSentence, int** matrix, int* iLen, int* jLen);

THOT_API void swAlignModel_close(void* swAlignModelHandle);

THOT_API bool giza_symmetr1(const char* lhsFileName, const char* rhsFileName, const char* outputFileName, bool transpose);

THOT_API bool phraseModel_generate(const char* alignmentFileName, int maxPhraseLength, const char* tableFileName);

#ifdef __cplusplus
}
#endif

#endif