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

THOT_API void decoder_close(void* decoderHandle);

THOT_API void* session_translate(void* sessionHandle, const wchar_t* sentence);

THOT_API void* session_translateInteractively(void* sessionHandle, const wchar_t* sentence);

THOT_API void* session_addStringToPrefix(void* sessionHandle, const wchar_t* addition);

THOT_API void* session_setPrefix(void* sessionHandle, const wchar_t* prefix);

THOT_API void session_trainSentencePair(void* decoderHandle, const wchar_t* sourceSentence, const wchar_t* targetSentence);

THOT_API void session_close(void* sessionHandle);

THOT_API const wchar_t* result_getTranslation(void* resultHandle);

THOT_API int result_getAlignedSourceWordIndex(void* resultHandle, int wordIndex);

THOT_API float result_getWordConfidence(void* resultHandle, int wordIndex);

THOT_API int result_getWordCount(void* resultHandle);

THOT_API void result_cleanup(void* resultHandle);

#ifdef __cplusplus
}
#endif

#endif