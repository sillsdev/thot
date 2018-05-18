// thot.cpp : Defines the exported functions for the DLL application.
//

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include "thot.h"

#include <SwModelInfo.h>
#include <PhraseModelInfo.h>
#include <LangModelInfo.h>
#include <_incrSwAligModel.h>
#include <_phraseBasedTransModel.h>
#include <_phrSwTransModel.h>
#include THOT_SMTMODEL_H
#include <BasePbTransModel.h>
#include <TranslationMetadata.h>
#include <IncrJelMerNgramLM.h>
#include <WbaIncrPhraseModel.h>
#include <WordPenaltyModel.h>
#include <multi_stack_decoder_rec.h>
#include <IncrHmmP0AligModel.h>
#include <MiraBleu.h>
#include <KbMiraLlWu.h>

#include <sstream>

extern "C"
{

struct SmtModelInfo
{
  SwModelInfo* swModelInfoPtr;
  PhraseModelInfo* phrModelInfoPtr;
  LangModelInfo* langModelInfoPtr;
  BasePbTransModel<SmtModel::Hypothesis>* smtModelPtr;
  BaseTranslationMetadata<SmtModel::HypScoreInfo>* trMetadataPtr;
  BaseScorer* scorerPtr;
  BaseLogLinWeightUpdater* llWeightUpdaterPtr;
  std::string lmFileName;
  std::string tmFileNamePrefix;
};

struct DecoderInfo
{
  SmtModelInfo* smtModelInfoPtr;
  BasePbTransModel<SmtModel::Hypothesis>* smtModelPtr;
  _stackDecoderRec<SmtModel>* stackDecoderPtr;
  BaseTranslationMetadata<SmtModel::HypScoreInfo>* trMetadataPtr;
};

struct LlWeightUpdaterInfo
{
  BaseScorer* baseScorerPtr;
  BaseLogLinWeightUpdater* llWeightUpdaterPtr;
};

struct WordGraphInfo
{
  std::string wordGraphStr;
  Score initialStateScore;
};

unsigned int copyString(const std::string& result,char* cstring,unsigned int capacity)
{
  if(cstring!=NULL)
  {
    unsigned int len=result.copy(cstring,capacity);
    if(len<capacity)
      cstring[len]='\0';
  }
  return result.length();
}

void* smtModel_create()
{
  SmtModelInfo* smtModelInfo=new SmtModelInfo;

  smtModelInfo->langModelInfoPtr=new LangModelInfo;
  smtModelInfo->phrModelInfoPtr=new PhraseModelInfo;
  smtModelInfo->swModelInfoPtr=new SwModelInfo;

  smtModelInfo->langModelInfoPtr->wpModelPtr=new WordPenaltyModel;
  smtModelInfo->langModelInfoPtr->lModelPtr=new IncrJelMerNgramLM;
  smtModelInfo->phrModelInfoPtr->invPbModelPtr=new WbaIncrPhraseModel;
  smtModelInfo->swModelInfoPtr->swAligModelPtrVec.push_back(new IncrHmmP0AligModel);
  smtModelInfo->swModelInfoPtr->invSwAligModelPtrVec.push_back(new IncrHmmP0AligModel);
  smtModelInfo->scorerPtr=new MiraBleu;
  smtModelInfo->llWeightUpdaterPtr=new KbMiraLlWu;
  smtModelInfo->trMetadataPtr=new TranslationMetadata<PhrScoreInfo>;

      // Link scorer to weight updater
  if(!smtModelInfo->llWeightUpdaterPtr->link_scorer(smtModelInfo->scorerPtr))
  {
    std::cerr<<"Error: Scorer class could not be linked to log-linear weight updater"<<std::endl;
    return NULL;
  }

      // Instantiate smt model
  smtModelInfo->smtModelPtr=new SmtModel;

      // Link pointers
  _phraseBasedTransModel<SmtModel::Hypothesis>* base_pbtm_ptr=dynamic_cast<_phraseBasedTransModel<SmtModel::Hypothesis>* >(smtModelInfo->smtModelPtr);
  if(base_pbtm_ptr)
  {
    base_pbtm_ptr->link_lm_info(smtModelInfo->langModelInfoPtr);
    base_pbtm_ptr->link_pm_info(smtModelInfo->phrModelInfoPtr);
  }
  _phrSwTransModel<SmtModel::Hypothesis>* base_pbswtm_ptr=dynamic_cast<_phrSwTransModel<SmtModel::Hypothesis>* >(smtModelInfo->smtModelPtr);
  if(base_pbswtm_ptr)
  {
    base_pbswtm_ptr->link_swm_info(smtModelInfo->swModelInfoPtr);
  }
  smtModelInfo->smtModelPtr->link_trans_metadata(smtModelInfo->trMetadataPtr);

  return smtModelInfo;
}

bool smtModel_loadTranslationModel(void* smtModelHandle,const char* tmFileNamePrefix)
{
  SmtModelInfo* smtModelInfo=static_cast<SmtModelInfo*>(smtModelHandle);
  _phraseBasedTransModel<SmtModel::Hypothesis>* phrbtm_ptr=dynamic_cast<_phraseBasedTransModel<SmtModel::Hypothesis>* >(smtModelInfo->smtModelPtr);
  if (phrbtm_ptr)
  {
    if (strcmp(smtModelInfo->tmFileNamePrefix.c_str(),tmFileNamePrefix)==0)
      return true;

    smtModelInfo->tmFileNamePrefix=tmFileNamePrefix;
    return phrbtm_ptr->loadAligModel(tmFileNamePrefix);
  }
  return false;
}

bool smtModel_loadLanguageModel(void* smtModelHandle,const char* lmFileName)
{
  SmtModelInfo* smtModelInfo=static_cast<SmtModelInfo*>(smtModelHandle);
  _phraseBasedTransModel<SmtModel::Hypothesis>* phrbtm_ptr=dynamic_cast<_phraseBasedTransModel<SmtModel::Hypothesis>* >(smtModelInfo->smtModelPtr);
  if (phrbtm_ptr)
  {
    if(strcmp(smtModelInfo->lmFileName.c_str(),lmFileName)==0)
      return true;

    smtModelInfo->lmFileName=lmFileName;
    return phrbtm_ptr->loadLangModel(lmFileName);
  }
  return false;
}

void smtModel_setNonMonotonicity(void* smtModelHandle,unsigned int nomon)
{
  SmtModelInfo* smtModelInfo=static_cast<SmtModelInfo*>(smtModelHandle);
  smtModelInfo->smtModelPtr->set_U_par(nomon);
}

void smtModel_setW(void* smtModelHandle,float w)
{
  SmtModelInfo* smtModelInfo=static_cast<SmtModelInfo*>(smtModelHandle);
  smtModelInfo->smtModelPtr->set_W_par(w);
}

void smtModel_setA(void* smtModelHandle,unsigned int a)
{
  SmtModelInfo* smtModelInfo=static_cast<SmtModelInfo*>(smtModelHandle);
  smtModelInfo->smtModelPtr->set_A_par(a);
}

void smtModel_setE(void* smtModelHandle,unsigned int e)
{
  SmtModelInfo* smtModelInfo=static_cast<SmtModelInfo*>(smtModelHandle);
  smtModelInfo->smtModelPtr->set_E_par(e);
}

void smtModel_setHeuristic(void* smtModelHandle,unsigned int heuristic)
{
  SmtModelInfo* smtModelInfo=static_cast<SmtModelInfo*>(smtModelHandle);
  smtModelInfo->smtModelPtr->setHeuristic(heuristic);
}

void smtModel_setOnlineTrainingParameters(void* smtModelHandle,unsigned int algorithm,unsigned int learningRatePolicy,float learnStepSize,
                                          unsigned int emIters,unsigned int e,unsigned int r)
{
  SmtModelInfo* smtModelInfo=static_cast<SmtModelInfo*>(smtModelHandle);
  OnlineTrainingPars otPars;
  otPars.onlineLearningAlgorithm=algorithm;
  otPars.learningRatePolicy=learningRatePolicy;
  otPars.learnStepSize=learnStepSize;
  otPars.emIters=emIters;
  otPars.E_par=e;
  otPars.R_par=r;
  smtModelInfo->smtModelPtr->setOnlineTrainingPars(otPars,0);
}

void smtModel_setWeights(void* smtModelHandle,const float* weights,unsigned int capacity)
{
  SmtModelInfo* smtModelInfo=static_cast<SmtModelInfo*>(smtModelHandle);
  std::vector<float> weightsVec;
  for(unsigned int i=0;i<capacity;++i)
    weightsVec.push_back(weights[i]);
  smtModelInfo->smtModelPtr->setWeights(weightsVec);
}

void* smtModel_getSingleWordAlignmentModel(void* smtModelHandle)
{
  SmtModelInfo* smtModelInfo=static_cast<SmtModelInfo*>(smtModelHandle);
  return smtModelInfo->swModelInfoPtr->swAligModelPtrVec[0];
}

void* smtModel_getInverseSingleWordAlignmentModel(void* smtModelHandle)
{
  SmtModelInfo* smtModelInfo=static_cast<SmtModelInfo*>(smtModelHandle);
  return smtModelInfo->swModelInfoPtr->invSwAligModelPtrVec[0];
}

bool smtModel_saveModels(void* smtModelHandle)
{
  SmtModelInfo* smtModelInfo=static_cast<SmtModelInfo*>(smtModelHandle);
  _phraseBasedTransModel<SmtModel::Hypothesis>* phrbtm_ptr=dynamic_cast<_phraseBasedTransModel<SmtModel::Hypothesis>* >(smtModelInfo->smtModelPtr);
  if (phrbtm_ptr)
  {
    if(!phrbtm_ptr->printAligModel(smtModelInfo->tmFileNamePrefix))
      return false;

    return phrbtm_ptr->printLangModel(smtModelInfo->lmFileName);
  }
  return false;
}

void smtModel_close(void* smtModelHandle)
{
  SmtModelInfo* smtModelInfo=static_cast<SmtModelInfo*>(smtModelHandle);

  smtModelInfo->smtModelPtr->clear();

  // Delete pointers
  delete smtModelInfo->langModelInfoPtr->lModelPtr;
  delete smtModelInfo->langModelInfoPtr->wpModelPtr;
  delete smtModelInfo->langModelInfoPtr;
  delete smtModelInfo->phrModelInfoPtr->invPbModelPtr;
  delete smtModelInfo->phrModelInfoPtr;
  delete smtModelInfo->swModelInfoPtr->swAligModelPtrVec[0];
  delete smtModelInfo->swModelInfoPtr->invSwAligModelPtrVec[0];
  delete smtModelInfo->swModelInfoPtr;
  delete smtModelInfo->smtModelPtr;
  delete smtModelInfo->llWeightUpdaterPtr;
  delete smtModelInfo->scorerPtr;
  delete smtModelInfo->trMetadataPtr;

  delete smtModelInfo;
}

void* decoder_create(void* smtModelHandle)
{
  SmtModelInfo* smtModelInfo=static_cast<SmtModelInfo*>(smtModelHandle);

  DecoderInfo* decoderInfo=new DecoderInfo;

  decoderInfo->smtModelInfoPtr=smtModelInfo;

  decoderInfo->stackDecoderPtr=new multi_stack_decoder_rec<SmtModel>;

  // Create statistical machine translation model instance (it is
  // cloned from the main one)
  BaseSmtModel<SmtModel::Hypothesis>* baseSmtModelPtr=smtModelInfo->smtModelPtr->clone();
  decoderInfo->smtModelPtr=dynamic_cast<BasePbTransModel<SmtModel::Hypothesis>* >(baseSmtModelPtr);

  decoderInfo->trMetadataPtr=new TranslationMetadata<PhrScoreInfo>;
  decoderInfo->smtModelPtr->link_trans_metadata(decoderInfo->trMetadataPtr);

  decoderInfo->stackDecoderPtr->link_smt_model(decoderInfo->smtModelPtr);

  decoderInfo->stackDecoderPtr->useBestScorePruning(true);

  return decoderInfo;
}

void decoder_setS(void* decoderHandle,unsigned int s)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);
  decoderInfo->stackDecoderPtr->set_S_par(s);
}

void decoder_setBreadthFirst(void* decoderHandle,bool breadthFirst)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);
  decoderInfo->stackDecoderPtr->set_breadthFirst(breadthFirst);
}

void decoder_setG(void* decoderHandle,unsigned int g)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);
  decoderInfo->stackDecoderPtr->set_G_par(g);
}

void decoder_close(void* decoderHandle)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);

  delete decoderInfo->smtModelPtr;
  delete decoderInfo->stackDecoderPtr;
  delete decoderInfo->trMetadataPtr;

  delete decoderInfo;
}

void* decoder_translate(void* decoderHandle,const char* sentence)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);

  TranslationData* result=new TranslationData;

  // Use translator
  SmtModel::Hypothesis hyp=decoderInfo->stackDecoderPtr->translate(sentence);

  std::vector<std::pair<PositionIndex, PositionIndex> > amatrix;
  // Obtain phrase alignment
  decoderInfo->smtModelPtr->aligMatrix(hyp,amatrix);
  decoderInfo->smtModelPtr->getPhraseAlignment(amatrix,result->sourceSegmentation,result->targetSegmentCuts);
  result->target=decoderInfo->smtModelPtr->getTransInPlainTextVec(hyp,result->targetUnknownWords);
  result->score=decoderInfo->smtModelPtr->getScoreForHyp(hyp);
  result->scoreComponents=decoderInfo->smtModelPtr->scoreCompsForHyp(hyp);

  return result;
}

unsigned int decoder_translateNBest(void* decoderHandle,unsigned int n,const char* sentence,void** results)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);

  // Enable word graph generation
  decoderInfo->stackDecoderPtr->enableWordGraph();

    // Use translator
  decoderInfo->stackDecoderPtr->translate(sentence);
  WordGraph* wg=decoderInfo->stackDecoderPtr->getWordGraphPtr();

  decoderInfo->stackDecoderPtr->disableWordGraph();

  std::vector<TranslationData> translations;
  wg->obtainNbestList(n,translations);

  for(unsigned int i=0;i<n && i<translations.size();++i)
    results[i]=new TranslationData(translations[i]);

  return translations.size();
}

void* decoder_getWordGraph(void* decoderHandle,const char* sentence)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);

  WordGraphInfo* result=new WordGraphInfo;

  decoderInfo->stackDecoderPtr->useBestScorePruning(false);

  // Enable word graph generation
  decoderInfo->stackDecoderPtr->enableWordGraph();

  // Use translator
  SmtModel::Hypothesis hyp=decoderInfo->stackDecoderPtr->translate(sentence);
  WordGraph* wg=decoderInfo->stackDecoderPtr->getWordGraphPtr();

  decoderInfo->stackDecoderPtr->disableWordGraph();

  decoderInfo->stackDecoderPtr->useBestScorePruning(true);

  if(decoderInfo->smtModelPtr->isComplete(hyp))
  {
    // Remove non-useful states from word-graph
    wg->obtainWgComposedOfUsefulStates();
    wg->orderArcsTopol();

    std::ostringstream outS;
    wg->print(outS,false);
    result->wordGraphStr=outS.str();
    result->initialStateScore=wg->getInitialStateScore();
  }
  else
  {
    result->wordGraphStr="";
    result->initialStateScore=0;
  }

  return result;
}

void* decoder_getBestPhraseAlignment(void* decoderHandle,const char* sentence,const char* translation)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);

  TranslationData* result=new TranslationData();
  SmtModel::Hypothesis hyp=decoderInfo->stackDecoderPtr->translateWithRef(sentence,translation);

  std::vector<std::pair<PositionIndex, PositionIndex> > amatrix;
  // Obtain phrase alignment
  decoderInfo->smtModelPtr->aligMatrix(hyp,amatrix);
  decoderInfo->smtModelPtr->getPhraseAlignment(amatrix,result->sourceSegmentation,result->targetSegmentCuts);
  result->target=decoderInfo->smtModelPtr->getTransInPlainTextVec(hyp,result->targetUnknownWords);
  result->score=decoderInfo->smtModelPtr->getScoreForHyp(hyp);
  result->scoreComponents=decoderInfo->smtModelPtr->scoreCompsForHyp(hyp);

  return result;
}

bool decoder_trainSentencePair(void* decoderHandle,const char* sourceSentence,const char* targetSentence,const int** matrix,unsigned int iLen,unsigned int jLen)
{
  DecoderInfo* decoderInfo=static_cast<DecoderInfo*>(decoderHandle);

  WordAligMatrix waMatrix(iLen,jLen);
  for(unsigned int i=0;i<iLen;i++)
  {
    for(unsigned int j=0;j<jLen;j++)
      waMatrix.setValue(i,j,matrix[i][j]);
  }

  // Obtain system translation
#ifdef THOT_ENABLE_UPDATE_LLWEIGHTS
  decoderInfo->stackDecoderPtr->enableWordGraph();
#endif

  SmtModel::Hypothesis hyp=decoderInfo->stackDecoderPtr->translate(sourceSentence);
  std::string sysSent=decoderInfo->smtModelPtr->getTransInPlainText(hyp);

  // Add sentence to word-predictor
  decoderInfo->smtModelInfoPtr->smtModelPtr->addSentenceToWordPred(StrProcUtils::stringToStringVector(targetSentence));

#ifdef THOT_ENABLE_UPDATE_LLWEIGHTS
  // Train log-linear weights

  // Retrieve pointer to wordgraph
  WordGraph* wgPtr=decoderInfo->stackDecoderPtr->getWordGraphPtr();
  decoderInfo->smtModelInfoPtr->smtModelPtr->updateLogLinearWeights(targetSentence,wgPtr);

  decoderInfo->stackDecoderPtr->disableWordGraph();
#endif

  // Train generative models
  return decoderInfo->smtModelInfoPtr->smtModelPtr->onlineTrainFeatsSentPair(sourceSentence,targetSentence,sysSent.c_str(),waMatrix);
}

unsigned int tdata_getTarget(void* dataHandle,char* target,unsigned int capacity)
{
  TranslationData* data=static_cast<TranslationData*>(dataHandle);
  return copyString(StrProcUtils::stringVectorToString(data->target),target,capacity);
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
    for(std::set<PositionIndex>::const_iterator it=data->targetUnknownWords.begin();it!=data->targetUnknownWords.end() && i<capacity;++it)
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

unsigned int wg_getString(void* wgHandle,char* wordGraphStr,unsigned int capacity)
{
  WordGraphInfo* wordGraph=static_cast<WordGraphInfo*>(wgHandle);
  return copyString(wordGraph->wordGraphStr,wordGraphStr,capacity);
}

double wg_getInitialStateScore(void* wgHandle)
{
  WordGraphInfo* wg = static_cast<WordGraphInfo*>(wgHandle);
  return wg->initialStateScore;
}

void wg_destroy(void* wgHandle)
{
  WordGraphInfo* wordGraph=static_cast<WordGraphInfo*>(wgHandle);
  delete wordGraph;
}

void* swAlignModel_create()
{
  return new IncrHmmP0AligModel;
}

void* swAlignModel_open(const char* prefFileName)
{
  BaseSwAligModel<PpInfo>* swAligModelPtr=new IncrHmmP0AligModel;
  if(swAligModelPtr->load(prefFileName)==THOT_ERROR)
  {
    delete swAligModelPtr;
    return NULL;
  }
  return swAligModelPtr;
}

unsigned int swAlignModel_getSourceWordCount(void* swAlignModelHandle)
{
  BaseSwAligModel<PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<PpInfo>*>(swAlignModelHandle);
  return swAligModelPtr->getSrcVocabSize();
}

unsigned int swAlignModel_getSourceWord(void* swAlignModelHandle,unsigned int index,char* wordStr,unsigned int capacity)
{
  BaseSwAligModel<PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<PpInfo>*>(swAlignModelHandle);
  return copyString(swAligModelPtr->wordIndexToSrcString(index),wordStr,capacity);
}

unsigned int swAlignModel_getTargetWordCount(void* swAlignModelHandle)
{
  BaseSwAligModel<PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<PpInfo>*>(swAlignModelHandle);
  return swAligModelPtr->getTrgVocabSize();
}

unsigned int swAlignModel_getTargetWord(void* swAlignModelHandle,unsigned int index,char* wordStr,unsigned int capacity)
{
  BaseSwAligModel<PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<PpInfo>*>(swAlignModelHandle);
  return copyString(swAligModelPtr->wordIndexToTrgString(index),wordStr,capacity);
}

void swAlignModel_addSentencePair(void* swAlignModelHandle,const char* sourceSentence,const char* targetSentence,const int** matrix,unsigned int iLen,unsigned int jLen)
{
  BaseSwAligModel<PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<PpInfo>*>(swAlignModelHandle);

  std::vector<std::string> source=StrProcUtils::stringToStringVector(sourceSentence);
  std::vector<std::string> target=StrProcUtils::stringToStringVector(targetSentence);
  WordAligMatrix waMatrix(iLen,jLen);
  for(unsigned int i=0;i<iLen;i++)
  {
    for(unsigned int j=0;j<jLen;j++)
      waMatrix.setValue(i,j,matrix[i][j]);
  }

  std::pair<unsigned int,unsigned int> pui;
  swAligModelPtr->addSentPair(source,target,1,waMatrix,pui);
  for(unsigned int j = 0;j<source.size();j++)
    swAligModelPtr->addSrcSymbol(source[j]);
  for(unsigned int j = 0;j<target.size();j++)
    swAligModelPtr->addTrgSymbol(target[j]);
}

void swAlignModel_train(void* swAlignModelHandle,unsigned int numIters)
{
  BaseSwAligModel<PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<PpInfo>*>(swAlignModelHandle);
  _incrSwAligModel<PpInfo>* _incrSwAligModelPtr=dynamic_cast<_incrSwAligModel<PpInfo>*>(swAligModelPtr);
  if(_incrSwAligModelPtr!=NULL)
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

float swAlignModel_getTranslationProbabilityByIndex(void* swAlignModelHandle,unsigned int srcWordIndex,unsigned int trgWordIndex)
{
  BaseSwAligModel<PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<PpInfo>*>(swAlignModelHandle);
  return swAligModelPtr->pts(srcWordIndex,trgWordIndex);
}

float swAlignModel_getAlignmentProbability(void* swAlignModelHandle,unsigned int prevI,unsigned int sLen,unsigned int i)
{
  BaseSwAligModel<PpInfo>* swAligModelPtr=static_cast<BaseSwAligModel<PpInfo>*>(swAlignModelHandle);
  _incrHmmAligModel* hmmSwAligModelPtr=dynamic_cast<_incrHmmAligModel*>(swAligModelPtr);
  if(hmmSwAligModelPtr!=NULL)
    return hmmSwAligModelPtr->aProb(prevI,sLen,i);
  return 0;
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
  if(alExt.open(lhsFileName)==THOT_ERROR)
    return false;
  alExt.symmetr1(rhsFileName,outputFileName,transpose);
  return true;
}

bool phraseModel_generate(const char* alignmentFileName,int maxPhraseLength,const char* tableFileName)
{
  WbaIncrPhraseModel phraseModel;
  PhraseExtractParameters phePars;
  phePars.maxTrgPhraseLength=maxPhraseLength;
  if(phraseModel.generateWbaIncrPhraseModel(alignmentFileName,phePars,false)==THOT_ERROR)
    return false;

  _incrPhraseModel* _incrPhraseModelPtr = dynamic_cast<_incrPhraseModel*>(&phraseModel);
  _incrPhraseModelPtr->printTTable(tableFileName);
  return true;
}

void* langModel_open(const char* prefFileName)
{
  BaseNgramLM<LM_State>* lmPtr=new IncrJelMerNgramLM;
  if(lmPtr->load(prefFileName)==THOT_ERROR)
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
  llwuInfo->baseScorerPtr=new MiraBleu;
  llwuInfo->llWeightUpdaterPtr=new KbMiraLlWu;

  llwuInfo->llWeightUpdaterPtr->link_scorer(llwuInfo->baseScorerPtr);
  return llwuInfo;
}

void llWeightUpdater_updateClosedCorpus(void* llWeightUpdaterHandle,const char** references,const char*** nblists,const double*** scoreComps,const unsigned int* nblistLens,
                                        float* weights,unsigned int numSents,unsigned int numWeights)
{
  LlWeightUpdaterInfo* llwuInfo=static_cast<LlWeightUpdaterInfo*>(llWeightUpdaterHandle);

  std::vector<std::string> refsVec;
  std::vector<std::vector<std::string> > nblistsVec;
  std::vector<std::vector<std::vector<double> > > scoreCompsVec;
  for(unsigned int i=0;i<numSents;++i)
  {
    refsVec.push_back(references[i]);
    std::vector<std::string> nblistVec;
    std::vector<std::vector<double> > nblistScoreCompsVec;
    for(unsigned int j=0;j<nblistLens[i];++j)
    {
      nblistVec.push_back(nblists[i][j]);
      std::vector<double> transScoreCompsVec;
      for(unsigned int k=0;k<numWeights;++k)
        transScoreCompsVec.push_back(scoreComps[i][j][k]);
      nblistScoreCompsVec.push_back(transScoreCompsVec);
    }
    nblistsVec.push_back(nblistVec);
    scoreCompsVec.push_back(nblistScoreCompsVec);
  }

  std::vector<double> curWeightsVec;
  for(unsigned int i=0;i<numWeights;++i)
    curWeightsVec.push_back(weights[i]);

  std::vector<double> newWeightsVec;
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

