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

int copyResult(const string& result,char* translation,int capacity)
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

void* decoder_getSingleWordAlignmentModel(void* decoderHandle)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);
  return &decoderInfo->decoder.swAligModel();
}

void* decoder_getInverseSingleWordAlignmentModel(void* decoderHandle)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);
  return &decoderInfo->decoder.invSwAligModel();
}

void decoder_close(void* decoderHandle)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);
  delete decoderInfo;
}

int session_translate(void* sessionHandle,const char* sentence,char* translation,int capacity,void** data)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  string result;
  TranslationData* tdata = new TranslationData();
  sessionInfo->decoder->translateSentence(sessionInfo->userId,sentence,result,*tdata);
  *data=tdata;
  return copyResult(result,translation,capacity);
}

int session_translateInteractively(void* sessionHandle,const char* sentence,char* translation,int capacity,void** data)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  string result;
  TranslationData* tdata = new TranslationData();
  sessionInfo->decoder->startCat(sessionInfo->userId,sentence,result,*tdata);
  *data=tdata;
  return copyResult(result,translation,capacity);
}

int session_addStringToPrefix(void* sessionHandle,const char* addition,char* translation,int capacity,void** data)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  RejectedWordsSet rejectedWords;
  string result;
  TranslationData* tdata = new TranslationData();
  sessionInfo->decoder->addStrToPref(sessionInfo->userId,addition,rejectedWords,result,*tdata);
  *data=tdata;
  return copyResult(result,translation,capacity);
}

int session_setPrefix(void* sessionHandle,const char* prefix,char* translation,int capacity,void** data)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  RejectedWordsSet rejectedWords;
  string result;
  TranslationData* tdata = new TranslationData();
  sessionInfo->decoder->setPref(sessionInfo->userId,prefix,rejectedWords,result,*tdata);
  *data=tdata;
  return copyResult(result,translation,capacity);
}

void session_trainSentencePair(void* sessionHandle,const char* sourceSentence,const char* targetSentence)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  sessionInfo->decoder->onlineTrainSentPair(sessionInfo->userId,sourceSentence,targetSentence);
}

void session_close(void* sessionHandle)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);
  sessionInfo->decoder->release_user_data(sessionInfo->userId);
  delete sessionInfo;
}

int tdata_getPhraseCount(void* dataHandle)
{
  TranslationData* data=static_cast<TranslationData*>(dataHandle);
  return data->sourceSegmentation.size();
}

int tdata_getSourceSegmentation(void* dataHandle,int** sourceSegmentation,int capacity)
{
  TranslationData* data=static_cast<TranslationData*>(dataHandle);
  if(sourceSegmentation!=NULL)
  {
    for(int i=0;i<capacity && i<data->sourceSegmentation.size();i++)
    {
      sourceSegmentation[i][0]=data->sourceSegmentation[i].first;
      sourceSegmentation[i][1]=data->sourceSegmentation[i].second;
    }
  }
  return data->sourceSegmentation.size();
}

int tdata_getTargetSegmentCuts(void* dataHandle,int* targetSegmentCuts,int capacity)
{
  TranslationData* data=static_cast<TranslationData*>(dataHandle);
  if(targetSegmentCuts!=NULL)
  {
    for(int i=0;i<capacity && i<data->targetSegmentCuts.size();i++)
      targetSegmentCuts[i]=data->targetSegmentCuts[i];
  }
  return data->targetSegmentCuts.size(); 
}

int tdata_getTargetUnknownWords(void* dataHandle,int* targetUnknownWords,int capacity)
{
  TranslationData* data=static_cast<TranslationData*>(dataHandle);
  if(targetUnknownWords!=NULL)
  {
    int i=0;
    for(set<PositionIndex>::const_iterator it=data->targetUnknownWords.begin();it!=data->targetUnknownWords.end() && i<capacity;++it)
    {
      targetUnknownWords[i]=*it;
      i++;
    }
  }
  return data->targetUnknownWords.size();
}

void tdata_destroy(void* dataHandle)
{
  TranslationData* data=static_cast<TranslationData*>(dataHandle);
  delete data;
}

void* swAlignModel_create()
{
  return new CURR_SWM_TYPE;
}

void* swAlignModel_open(const char* prefFileName)
{
  BaseSwAligModel<CURR_SWM_TYPE::PpInfo>* swAligModelPtr=new CURR_SWM_TYPE;
  if(swAligModelPtr->load(prefFileName)==ERROR)
  {
    delete swAligModelPtr;
    return NULL;
  }
  return swAligModelPtr;
}

void swAlignModel_addSentencePair(void* swAlignModelHandle,const char* sourceSentence,const char* targetSentence)
{
  BaseSwAligModel<CURR_SWM_TYPE::PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<CURR_SWM_TYPE::PpInfo>*>(swAlignModelHandle);

  Vector<std::string> source=StrProcUtils::stringToStringVector(sourceSentence);
  Vector<std::string> target=StrProcUtils::stringToStringVector(targetSentence);
  pair<unsigned int,unsigned int> pui;
  swAligModelPtr->addSentPair(source,target,1,pui);
  for(int j = 0;j<source.size();j++)
    swAligModelPtr->addSrcSymbol(source[j],1);
  for(int j = 0;j<target.size();j++)
    swAligModelPtr->addTrgSymbol(target[j],1);
}

void swAlignModel_train(void* swAlignModelHandle,int numIters)
{
  BaseSwAligModel<CURR_SWM_TYPE::PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<CURR_SWM_TYPE::PpInfo>*>(swAlignModelHandle);
  _incrSwAligModel<CURR_SWM_TYPE::PpInfo>* _incrSwAligModelPtr=dynamic_cast<_incrSwAligModel<CURR_SWM_TYPE::PpInfo>*>(swAligModelPtr);
  if(_incrSwAligModelPtr != NULL)
  {
    for(int i=0;i<numIters;i++)
      _incrSwAligModelPtr->efficientBatchTrainingForAllSents();
  }
  else
  {
    for(int i=0;i<numIters;i++)
      swAligModelPtr->trainAllSents();
  }
}

void swAlignModel_save(void* swAlignModelHandle,const char* prefFileName)
{
  BaseSwAligModel<CURR_SWM_TYPE::PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<CURR_SWM_TYPE::PpInfo>*>(swAlignModelHandle);
  swAligModelPtr->print(prefFileName);
}

float swAlignModel_getTranslationProbability(void* swAlignModelHandle,const char* srcWord,const char* trgWord)
{
  BaseSwAligModel<CURR_SWM_TYPE::PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<CURR_SWM_TYPE::PpInfo>*>(swAlignModelHandle);
  WordIndex srcWordIndex=swAligModelPtr->stringToSrcWordIndex(srcWord);
  WordIndex trgWordIndex=swAligModelPtr->stringToTrgWordIndex(trgWord);
  return swAligModelPtr->pts(srcWordIndex,trgWordIndex);
}

float swAlignModel_getBestAlignment(void* swAlignModelHandle,const char* sourceSentence,const char* targetSentence,int** matrix,int* iLen,int* jLen)
{
  BaseSwAligModel<CURR_SWM_TYPE::PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<CURR_SWM_TYPE::PpInfo>*>(swAlignModelHandle);
  WordAligMatrix waMatrix;
  LgProb prob = swAligModelPtr->obtainBestAlignmentChar(sourceSentence,targetSentence,waMatrix);
  for(int i=0;i<*iLen;i++)
  {
    for(int j=0;j<*jLen;j++)
      matrix[i][j]=waMatrix.getValue(i,j);
  }
  *iLen=waMatrix.get_I();
  *jLen=waMatrix.get_J();
  return prob;
}

void swAlignModel_close(void* swAlignModelHandle)
{
  BaseSwAligModel<CURR_SWM_TYPE::PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<CURR_SWM_TYPE::PpInfo>*>(swAlignModelHandle);
  delete swAligModelPtr;
}

bool giza_symmetr1(const char* lhsFileName,const char* rhsFileName,const char* outputFileName,bool transpose)
{
  AlignmentExtractor alExt;
  if(alExt.open(lhsFileName)==ERROR)
    return false;
  alExt.symmetr1(rhsFileName,outputFileName,transpose);
  return true;
}

bool phraseModel_generate(const char* alignmentFileName,int maxPhraseLength,const char* tableFileName)
{
  WbaIncrPhraseModel wbaIncrPhraseModel;
  PhraseExtractParameters phePars;
  phePars.maxTrgPhraseLength=maxPhraseLength;
  if(wbaIncrPhraseModel.generateWbaIncrPhraseModel(alignmentFileName,phePars,false)==ERROR)
    return false;

  wbaIncrPhraseModel.printTTable(tableFileName);
  return true;
}

void* langModel_open(const char* prefFileName)
{
  BaseNgramLM<THOT_CURR_LM_TYPE::LM_State>* lmPtr=new THOT_CURR_LM_TYPE;
  if(lmPtr->load(prefFileName)==ERROR)
  {
    delete lmPtr;
    return NULL;
  }
  return lmPtr;
}

float langModel_getSentenceProbability(void* lmHandle,const char* sentence)
{
  BaseNgramLM<THOT_CURR_LM_TYPE::LM_State>* lmPtr=static_cast<BaseNgramLM<THOT_CURR_LM_TYPE::LM_State>*>(lmHandle);
  return lmPtr->getSentenceLog10ProbStr(StrProcUtils::stringToStringVector(sentence));
}

void langModel_close(void* lmHandle)
{
  BaseNgramLM<THOT_CURR_LM_TYPE::LM_State>* lmPtr=static_cast<BaseNgramLM<THOT_CURR_LM_TYPE::LM_State>*>(lmHandle);
  delete lmPtr;
}

}

