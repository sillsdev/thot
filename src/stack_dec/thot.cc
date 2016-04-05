// thot.cpp : Defines the exported functions for the DLL application.
//

#include "thot.h"
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
  string targetSentence;
  Vector<pair<unsigned int,float> > alignment;
};

TranslationResult* CreateResult(ThotDecoder* decoder, const string& source, const string& target)
{
  TranslationResult* result=new TranslationResult();
  result->targetSentence=target;
  decoder->getWordAlignment(source.c_str(),target.c_str(),result->alignment);
  return result;
}

void* decoder_open(const char* cfgFileName)
{
  DecoderInfo* decoderInfo=new DecoderInfo();
  if(decoderInfo->decoder.initUsingCfgFile(cfgFileName,decoderInfo->userParams,0)==ERROR)
  {
    delete decoderInfo;
    return NULL;
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
    return NULL;

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

void* session_translate(void* sessionHandle, const char* sentence)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  string result;
  sessionInfo->decoder->translateSentence(sessionInfo->userId,sentence,result);

  return CreateResult(sessionInfo->decoder,sentence,result);
}

void* session_translateInteractively(void* sessionHandle, const char* sentence)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  sessionInfo->imtSentence=sentence;

  string result;
  sessionInfo->decoder->startCat(sessionInfo->userId,sessionInfo->imtSentence.c_str(),result);

  return CreateResult(sessionInfo->decoder,sessionInfo->imtSentence,result);
}

void* session_addStringToPrefix(void* sessionHandle, const char* addition)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  RejectedWordsSet rejectedWords;
  string result;
  sessionInfo->decoder->addStrToPref(sessionInfo->userId,addition,rejectedWords,result);

  return CreateResult(sessionInfo->decoder,sessionInfo->imtSentence,result);
}

void* session_setPrefix(void* sessionHandle, const char* prefix)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  RejectedWordsSet rejectedWords;
  string result;
  sessionInfo->decoder->setPref(sessionInfo->userId,prefix,rejectedWords,result);

  return CreateResult(sessionInfo->decoder,sessionInfo->imtSentence,result);
}

void session_trainSentencePair(void* sessionHandle, const char* sourceSentence, const char* targetSentence)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  sessionInfo->decoder->onlineTrainSentPair(sessionInfo->userId,sourceSentence,targetSentence);
}

void session_close(void* sessionHandle)
{
  SessionInfo* sessionInfo = static_cast<SessionInfo*>(sessionHandle);
  sessionInfo->decoder->release_user_data(sessionInfo->userId);
  delete sessionInfo;
}

const char* result_getTranslation(void* resultHandle)
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

