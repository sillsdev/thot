// thot.cpp : Defines the exported functions for the DLL application.
//

#include "thot.h"
#include "ThotDecoder.h"
#include "StandardClasses.h"
#include <_incrSwAligModel.h>

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

struct LlWeightUpdaterInfo
{
  BaseScorer* baseScorerPtr;
  BaseLogLinWeightUpdater* llWeightUpdaterPtr;
};

unsigned int copyResult(const string& result,char* translation,unsigned int capacity)
{
  if(translation!=NULL)
  {
    unsigned int len=result.copy(translation,capacity);
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
  return decoderInfo->decoder.swModelInfoPtr()->swAligModelPtr;
}

void* decoder_getInverseSingleWordAlignmentModel(void* decoderHandle)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);
  return decoderInfo->decoder.swModelInfoPtr()->invSwAligModelPtr;
}

void decoder_setLlWeights(void* decoderHandle,const float* weights,unsigned int capacity)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);
  Vector<float> tmw;
  for(unsigned int i=0;i<capacity;++i)
    tmw.push_back(weights[i]);
  decoderInfo->decoder.set_tmw(tmw);
}

void decoder_close(void* decoderHandle)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);
  delete decoderInfo;
}

void* session_translate(void* sessionHandle,const char* sentence)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  TranslationData* result=new TranslationData();
  if(sessionInfo->decoder->translateSentence(sessionInfo->userId,sentence,*result)==OK)
    return result;

  delete result;
  return NULL;
}

unsigned int session_translateNBest(void* sessionHandle,unsigned int n,const char* sentence,void** results)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  Vector<TranslationData> translations;
  if(sessionInfo->decoder->translateSentence(sessionInfo->userId,n,sentence,translations)==OK)
  {
    for(unsigned int i=0;i<n && i<translations.size();++i)
      results[i]=new TranslationData(translations[i]);

    return translations.size();
  }

  return 0;
}

void* session_getBestPhraseAlignment(void* sessionHandle,const char* sentence,const char* translation)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  TranslationData* result=new TranslationData();
  if(sessionInfo->decoder->sentPairBestAlignment(sessionInfo->userId,sentence,translation,*result)==OK)
    return result;

  delete result;
  return NULL;
}

void* session_translateInteractively(void* sessionHandle,const char* sentence)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  TranslationData* result=new TranslationData();
  if(sessionInfo->decoder->startCat(sessionInfo->userId,sentence,*result)==OK)
    return result;

  delete result;
  return NULL;
}

void* session_addStringToPrefix(void* sessionHandle,const char* addition)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  RejectedWordsSet rejectedWords;
  TranslationData* result=new TranslationData();
  sessionInfo->decoder->addStrToPref(sessionInfo->userId,addition,rejectedWords,*result);
  return result;
}

void* session_setPrefix(void* sessionHandle,const char* prefix)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  RejectedWordsSet rejectedWords;
  TranslationData* result=new TranslationData();
  sessionInfo->decoder->setPref(sessionInfo->userId,prefix,rejectedWords,*result);
  return result;
}

void session_trainSentencePair(void* sessionHandle,const char* sourceSentence,const char* targetSentence,const int** matrix,unsigned int iLen,unsigned int jLen)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);

  WordAligMatrix waMatrix(iLen,jLen);
  for(unsigned int i=0;i<iLen;i++)
  {
    for(unsigned int j=0;j<jLen;j++)
      waMatrix.setValue(i,j,matrix[i][j]);
  }

  sessionInfo->decoder->onlineTrainSentPair(sessionInfo->userId,sourceSentence,targetSentence,waMatrix);
}

void session_close(void* sessionHandle)
{
  SessionInfo* sessionInfo=static_cast<SessionInfo*>(sessionHandle);
  sessionInfo->decoder->release_user_data(sessionInfo->userId);
  delete sessionInfo;
}

unsigned int tdata_getTarget(void* dataHandle,char* target,unsigned int capacity)
{
  TranslationData* data=static_cast<TranslationData*>(dataHandle);
  return copyResult(StrProcUtils::stringVectorToString(data->target),target,capacity);
}

unsigned int tdata_getPhraseCount(void* dataHandle)
{
  TranslationData* data=static_cast<TranslationData*>(dataHandle);
  return data->sourceSegmentation.size();
}

unsigned int tdata_getSourceSegmentation(void* dataHandle,unsigned int** sourceSegmentation,unsigned int capacity)
{
  TranslationData* data=static_cast<TranslationData*>(dataHandle);
  if(sourceSegmentation!=NULL)
  {
    for(unsigned int i=0;i<capacity && i<data->sourceSegmentation.size();i++)
    {
      sourceSegmentation[i][0]=data->sourceSegmentation[i].first;
      sourceSegmentation[i][1]=data->sourceSegmentation[i].second;
    }
  }
  return data->sourceSegmentation.size();
}

unsigned int tdata_getTargetSegmentCuts(void* dataHandle,unsigned int* targetSegmentCuts,unsigned int capacity)
{
  TranslationData* data=static_cast<TranslationData*>(dataHandle);
  if(targetSegmentCuts!=NULL)
  {
    for(unsigned int i=0;i<capacity && i<data->targetSegmentCuts.size();i++)
      targetSegmentCuts[i]=data->targetSegmentCuts[i];
  }
  return data->targetSegmentCuts.size(); 
}

unsigned int tdata_getTargetUnknownWords(void* dataHandle,unsigned int* targetUnknownWords,unsigned int capacity)
{
  TranslationData* data=static_cast<TranslationData*>(dataHandle);
  if(targetUnknownWords!=NULL)
  {
    unsigned int i=0;
    for(set<PositionIndex>::const_iterator it=data->targetUnknownWords.begin();it!=data->targetUnknownWords.end() && i<capacity;++it)
    {
      targetUnknownWords[i]=*it;
      i++;
    }
  }
  return data->targetUnknownWords.size();
}

double tdata_getScore(void* dataHandle)
{
  TranslationData* data=static_cast<TranslationData*>(dataHandle);
  return data->score;
}

unsigned int tdata_getScoreComponents(void* dataHandle,double* scoreComps,unsigned int capacity)
{
  TranslationData* data=static_cast<TranslationData*>(dataHandle);
  for(unsigned int i=0;i<capacity && i<data->scoreComponents.size();i++)
    scoreComps[i]=data->scoreComponents[i];
  return data->scoreComponents.size();
}

void tdata_destroy(void* dataHandle)
{
  TranslationData* data=static_cast<TranslationData*>(dataHandle);
  delete data;
}

void* swAlignModel_create()
{
  return new SW_ALIG_MODEL;
}

void* swAlignModel_open(const char* prefFileName)
{
  BaseSwAligModel<PpInfo>* swAligModelPtr=new SW_ALIG_MODEL;
  if(swAligModelPtr->load(prefFileName)==ERROR)
  {
    delete swAligModelPtr;
    return NULL;
  }
  return swAligModelPtr;
}

void swAlignModel_addSentencePair(void* swAlignModelHandle,const char* sourceSentence,const char* targetSentence,const int** matrix,unsigned int iLen,unsigned int jLen)
{
  BaseSwAligModel<PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<PpInfo>*>(swAlignModelHandle);

  Vector<std::string> source=StrProcUtils::stringToStringVector(sourceSentence);
  Vector<std::string> target=StrProcUtils::stringToStringVector(targetSentence);
  WordAligMatrix waMatrix(iLen,jLen);
  for(unsigned int i=0;i<iLen;i++)
  {
    for(unsigned int j=0;j<jLen;j++)
      waMatrix.setValue(i,j,matrix[i][j]);
  }

  pair<unsigned int,unsigned int> pui;
  swAligModelPtr->addSentPair(source,target,1,waMatrix,pui);
  for(unsigned int j = 0;j<source.size();j++)
    swAligModelPtr->addSrcSymbol(source[j],1);
  for(unsigned int j = 0;j<target.size();j++)
    swAligModelPtr->addTrgSymbol(target[j],1);
}

void swAlignModel_train(void* swAlignModelHandle,unsigned int numIters)
{
  BaseSwAligModel<PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<PpInfo>*>(swAlignModelHandle);
  _incrSwAligModel<PpInfo>* _incrSwAligModelPtr=dynamic_cast<_incrSwAligModel<PpInfo>*>(swAligModelPtr);
  if(_incrSwAligModelPtr != NULL)
  {
    for(unsigned int i=0;i<numIters;i++)
      _incrSwAligModelPtr->efficientBatchTrainingForAllSents();
  }
  else
  {
    for(unsigned int i=0;i<numIters;i++)
      swAligModelPtr->trainAllSents();
  }
}

void swAlignModel_save(void* swAlignModelHandle,const char* prefFileName)
{
  BaseSwAligModel<PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<PpInfo>*>(swAlignModelHandle);
  swAligModelPtr->print(prefFileName);
}

float swAlignModel_getTranslationProbability(void* swAlignModelHandle,const char* srcWord,const char* trgWord)
{
  BaseSwAligModel<PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<PpInfo>*>(swAlignModelHandle);
  WordIndex srcWordIndex=swAligModelPtr->stringToSrcWordIndex(srcWord);
  WordIndex trgWordIndex=swAligModelPtr->stringToTrgWordIndex(trgWord);
  return swAligModelPtr->pts(srcWordIndex,trgWordIndex);
}

float swAlignModel_getBestAlignment(void* swAlignModelHandle,const char* sourceSentence,const char* targetSentence,int** matrix,unsigned int* iLen,unsigned int* jLen)
{
  BaseSwAligModel<PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<PpInfo>*>(swAlignModelHandle);
  WordAligMatrix waMatrix(*iLen,*jLen);
  for(unsigned int i=0;i<*iLen;i++)
  {
    for(unsigned int j=0;j<*jLen;j++)
      waMatrix.setValue(i,j,matrix[i][j]);
  }

  LgProb prob=swAligModelPtr->obtainBestAlignmentChar(sourceSentence,targetSentence,waMatrix);
  for(unsigned int i=0;i<*iLen;i++)
  {
    for(unsigned int j=0;j<*jLen;j++)
      matrix[i][j]=waMatrix.getValue(i,j);
  }
  *iLen=waMatrix.get_I();
  *jLen=waMatrix.get_J();
  return prob;
}

void swAlignModel_close(void* swAlignModelHandle)
{
  BaseSwAligModel<PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<PpInfo>*>(swAlignModelHandle);
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
  PHRASE_MODEL phraseModel;
  PhraseExtractParameters phePars;
  phePars.maxTrgPhraseLength=maxPhraseLength;
  if(phraseModel.generateWbaIncrPhraseModel(alignmentFileName,phePars,false)==ERROR)
    return false;

  phraseModel.printTTable(tableFileName);
  return true;
}

void* langModel_open(const char* prefFileName)
{
  BaseNgramLM<LM_State>* lmPtr=new NGRAM_LM;
  if(lmPtr->load(prefFileName)==ERROR)
  {
    delete lmPtr;
    return NULL;
  }
  return lmPtr;
}

float langModel_getSentenceProbability(void* lmHandle,const char* sentence)
{
  BaseNgramLM<LM_State>* lmPtr=static_cast<BaseNgramLM<LM_State>*>(lmHandle);
  return lmPtr->getSentenceLog10ProbStr(StrProcUtils::stringToStringVector(sentence));
}

void langModel_close(void* lmHandle)
{
  BaseNgramLM<LM_State>* lmPtr=static_cast<BaseNgramLM<LM_State>*>(lmHandle);
  delete lmPtr;
}

void* llWeightUpdater_create()
{
  LlWeightUpdaterInfo* llwuInfo=new LlWeightUpdaterInfo;
  llwuInfo->baseScorerPtr=new SCORER;
  llwuInfo->llWeightUpdaterPtr=new LL_WEIGHT_UPDATER;

  llwuInfo->llWeightUpdaterPtr->link_scorer(llwuInfo->baseScorerPtr);
  return llwuInfo;
}

void llWeightUpdater_updateClosedCorpus(void* llWeightUpdaterHandle,const char** references,const char*** nblists,const double*** scoreComps,const unsigned int* nblistLens,
                                        double* weights,unsigned int numSents,unsigned int numWeights)
{
  LlWeightUpdaterInfo* llwuInfo=static_cast<LlWeightUpdaterInfo*>(llWeightUpdaterHandle);

  Vector<std::string> refsVec;
  Vector<Vector<std::string> > nblistsVec;
  Vector<Vector<Vector<double> > > scoreCompsVec;
  for(unsigned int i=0;i<numSents;++i)
  {
    refsVec.push_back(references[i]);
    Vector<std::string> nblistVec;
    Vector<Vector<double> > nblistScoreCompsVec;
    for(unsigned int j=0;j<nblistLens[i];++j)
    {
      nblistVec.push_back(nblists[i][j]);
      Vector<double> transScoreCompsVec;
      for(unsigned int k=0;k<numWeights;++k)
        transScoreCompsVec.push_back(scoreComps[i][j][k]);
      nblistScoreCompsVec.push_back(transScoreCompsVec);
    }
    nblistsVec.push_back(nblistVec);
    scoreCompsVec.push_back(nblistScoreCompsVec);
  }

  Vector<double> curWeightsVec;
  for(unsigned int i=0;i<numWeights;++i)
    curWeightsVec.push_back(weights[i]);

  Vector<double> newWeightsVec;
  llwuInfo->llWeightUpdaterPtr->updateClosedCorpus(refsVec,nblistsVec,scoreCompsVec,curWeightsVec,newWeightsVec);

  for(unsigned int i=0;i<numWeights;++i)
    weights[i]=newWeightsVec[i];
}

void llWeightUpdater_close(void* llWeightUpdaterHandle)
{
  LlWeightUpdaterInfo* llwuInfo=static_cast<LlWeightUpdaterInfo*>(llWeightUpdaterHandle);
  delete llwuInfo->llWeightUpdaterPtr;
  delete llwuInfo->baseScorerPtr;
  delete llwuInfo;
}

}

