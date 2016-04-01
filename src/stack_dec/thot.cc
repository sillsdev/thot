// thot.cpp : Defines the exported functions for the DLL application.
//

#include "thot.h"
#include <codecvt>
#include "ThotDecoder.h"

extern "C"
{

struct DecoderInfo
{
  ThotDecoder decoder;
  ThotDecoderUserPars userParams;
};

struct SessionInfo
{
  int userId;
  ThotDecoder* decoder;
  string imtSentence;
};

struct TranslationResult
{
  wstring targetSentence;
  Vector<pair<unsigned int,float>> alignment;
};

TranslationResult* CreateResult(ThotDecoder* decoder, wstring_convert<codecvt_utf8<wchar_t>>& utf8Convert, const string& utf8Source, const string& utf8Target)
{
  TranslationResult* result=new TranslationResult();
  result->targetSentence=utf8Convert.from_bytes(utf8Target);
  decoder->getWordAlignment(utf8Source.c_str(),utf8Target.c_str(),result->alignment);
  return result;
}

void* decoder_open(const char* cfgFileName)
{
  DecoderInfo* decoderInfo=new DecoderInfo();
  if(decoderInfo->decoder.initUsingCfgFile(cfgFileName,decoderInfo->userParams,0)==ERROR)
  {
    delete decoderInfo;
    return nullptr;
  }

  return decoderInfo;
}

void* decoder_openSession(void* decoderHandle)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);
  int userId=0;
  while(!decoderInfo->decoder.user_id_new(userId))
    userId++;
  if(decoderInfo->decoder.initUserPars(userId,decoderInfo->userParams,0)==ERROR)
    return nullptr;

  SessionInfo* sessionInfo=new SessionInfo();
  sessionInfo->userId=userId;
  sessionInfo->decoder=&decoderInfo->decoder;
  return sessionInfo;
}

void decoder_saveModels(void* decoderHandle)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);
  decoderInfo->decoder.printModels();
}

void decoder_close(void* decoderHandle)
{
  delete decoderHandle;
}

void* session_translate(void* sessionHandle, const wchar_t* sentence)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  wstring_convert<codecvt_utf8<wchar_t>> utf8Convert;
  string utf8Sentence=utf8Convert.to_bytes(sentence);

  string utf8Result;
  sessionInfo->decoder->translateSentence(sessionInfo->userId,utf8Sentence.c_str(),utf8Result);

  return CreateResult(sessionInfo->decoder,utf8Convert,utf8Sentence,utf8Result);
}

void* session_translateInteractively(void* sessionHandle, const wchar_t* sentence)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  wstring_convert<codecvt_utf8<wchar_t>> utf8Convert;
  sessionInfo->imtSentence=utf8Convert.to_bytes(sentence);

  string utf8Result;
  sessionInfo->decoder->startCat(sessionInfo->userId,sessionInfo->imtSentence.c_str(),utf8Result);

  return CreateResult(sessionInfo->decoder,utf8Convert,sessionInfo->imtSentence,utf8Result);
}

void* session_addStringToPrefix(void* sessionHandle, const wchar_t* addition)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  wstring_convert<codecvt_utf8<wchar_t>> utf8Convert;
  string utf8Addition=utf8Convert.to_bytes(addition);

  RejectedWordsSet rejectedWords;
  string utf8Result;
  sessionInfo->decoder->addStrToPref(sessionInfo->userId,utf8Addition.c_str(),rejectedWords,utf8Result);

  return CreateResult(sessionInfo->decoder,utf8Convert,sessionInfo->imtSentence,utf8Result);
}

void* session_setPrefix(void* sessionHandle, const wchar_t* prefix)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  wstring_convert<codecvt_utf8<wchar_t>> utf8Convert;
  string utf8Prefix=utf8Convert.to_bytes(prefix);

  RejectedWordsSet rejectedWords;
  string utf8Result;
  sessionInfo->decoder->setPref(sessionInfo->userId,utf8Prefix.c_str(),rejectedWords,utf8Result);

  return CreateResult(sessionInfo->decoder,utf8Convert,sessionInfo->imtSentence,utf8Result);
}

void session_trainSentencePair(void* sessionHandle, const wchar_t* sourceSentence, const wchar_t* targetSentence)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  wstring_convert<codecvt_utf8<wchar_t>> utf8_conv;
  string utf8SourceSentence=utf8_conv.to_bytes(sourceSentence);
  string utf8TargetSentence=utf8_conv.to_bytes(targetSentence);

  sessionInfo->decoder->onlineTrainSentPair(sessionInfo->userId,utf8SourceSentence.c_str(),utf8TargetSentence.c_str());
}

void session_close(void* sessionHandle)
{
  SessionInfo* sessionInfo = static_cast<SessionInfo*>(sessionHandle);
  sessionInfo->decoder->release_user_data(sessionInfo->userId);
  delete sessionInfo;
}

const wchar_t* result_getTranslation(void* resultHandle)
{
  TranslationResult* result=static_cast<TranslationResult*>(resultHandle);
  return result->targetSentence.c_str();
}

int result_getAlignedSourceWordIndex(void* resultHandle, int wordIndex)
{
  TranslationResult* result = static_cast<TranslationResult*>(resultHandle);
  return result->alignment[wordIndex].first;
}

float result_getWordConfidence(void* resultHandle, int wordIndex)
{
  TranslationResult* result=static_cast<TranslationResult*>(resultHandle);
  return result->alignment[wordIndex].second;
}

int result_getWordCount(void* resultHandle)
{
  TranslationResult* result=static_cast<TranslationResult*>(resultHandle);
  return result->alignment.size();
}

void result_cleanup(void* resultHandle)
{
  delete resultHandle;
}

}

