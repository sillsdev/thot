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

THOT_API float decoder_getWordConfidence(void* decoderHandle, const char* srcWord, const char* trgWord);

THOT_API void decoder_close(void* decoderHandle);

THOT_API int session_translate(void* sessionHandle, const char* sentence, char* translation, int capacity);

THOT_API int session_translateInteractively(void* sessionHandle, const char* sentence, char* translation, int capacity);

THOT_API int session_addStringToPrefix(void* sessionHandle, const char* addition, char* translation, int capacity);

THOT_API int session_setPrefix(void* sessionHandle, const char* prefix, char* translation, int capacity);

THOT_API void session_trainSentencePair(void* decoderHandle, const char* sourceSentence, const char* targetSentence);

THOT_API void session_close(void* sessionHandle);

#ifdef __cplusplus
}
#endif

#endif