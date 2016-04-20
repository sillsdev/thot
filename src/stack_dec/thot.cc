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
};

int copyResult(const string& result, char* translation, int capacity)
{
  if(translation!=NULL)
  {
    int len=result.copy(translation,capacity);
    if(len<capacity)
      translation[len]='\0';
  }
  return result.length();
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

float decoder_getTranslationProbability(void* decoderHandle, const char* srcWord, const char* trgWord)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);
  return decoderInfo->decoder.getTranslationProbability(srcWord, trgWord);
}

int decoder_getBestAlignment(void* decoderHandle, const char* sourceSentence, const char* targetSentence, int* alignment, int capacity)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);
  Vector<PositionIndex> indices;
  decoderInfo->decoder.getBestAlignment(sourceSentence,targetSentence,indices);
  if (alignment!=NULL)
  {
    for (int i=0;i<indices.size() || i<capacity;i++)
      alignment[i]=indices[i];
  }
  return indices.size();
}

void decoder_close(void* decoderHandle)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);
  delete decoderInfo;
}

int session_translate(void* sessionHandle, const char* sentence, char* translation, int capacity)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  string result;
  sessionInfo->decoder->translateSentence(sessionInfo->userId,sentence,result);
  return copyResult(result,translation,capacity);
}

int session_translateInteractively(void* sessionHandle, const char* sentence, char* translation, int capacity)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  string result;
  sessionInfo->decoder->startCat(sessionInfo->userId,sentence,result);
  return copyResult(result,translation,capacity);
}

int session_addStringToPrefix(void* sessionHandle, const char* addition, char* translation, int capacity)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  RejectedWordsSet rejectedWords;
  string result;
  sessionInfo->decoder->addStrToPref(sessionInfo->userId,addition,rejectedWords,result);
  return copyResult(result,translation,capacity);
}

int session_setPrefix(void* sessionHandle, const char* prefix, char* translation, int capacity)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  RejectedWordsSet rejectedWords;
  string result;
  sessionInfo->decoder->setPref(sessionInfo->userId,prefix,rejectedWords,result);
  return copyResult(result,translation,capacity);
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

}

