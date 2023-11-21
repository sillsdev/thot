// thot.cpp : Defines the exported functions for the DLL application.
//

#include "thot.h"

#include "incr_models/IncrJelMerNgramLM.h"
#include "incr_models/WordPenaltyModel.h"
#include "phrase_models/WbaIncrPhraseModel.h"
#include "stack_dec/BasePbTransModel.h"
#include "stack_dec/KbMiraLlWu.h"
#include "stack_dec/LangModelInfo.h"
#include "stack_dec/MiraBleu.h"
#include "stack_dec/PhrLocalSwLiTm.h"
#include "stack_dec/PhraseModelInfo.h"
#include "stack_dec/SwModelInfo.h"
#include "stack_dec/TranslationMetadata.h"
#include "stack_dec/_phrSwTransModel.h"
#include "stack_dec/_phraseBasedTransModel.h"
#include "stack_dec/multi_stack_decoder_rec.h"
#include "sw_models/FastAlignModel.h"
#include "sw_models/HmmAlignmentModel.h"
#include "sw_models/Ibm1AlignmentModel.h"
#include "sw_models/Ibm2AlignmentModel.h"
#include "sw_models/Ibm3AlignmentModel.h"
#include "sw_models/Ibm4AlignmentModel.h"
#include "sw_models/IncrAlignmentModel.h"
#include "sw_models/IncrHmmAlignmentModel.h"
#include "sw_models/IncrIbm1AlignmentModel.h"
#include "sw_models/IncrIbm2AlignmentModel.h"

#include <memory>
#include <sstream>

struct SmtModelInfo
{
  std::unique_ptr<PhrLocalSwLiTm> smtModel{};
  std::string lmFileName;
  std::string tmFileNamePrefix;
};

struct WordGraphInfo
{
  std::string wordGraphStr;
  Score initialStateScore{};
};

unsigned int copyString(const std::string& result, char* cstring, unsigned int capacity)
{
  if (cstring != NULL)
  {
    size_t len = result.copy(cstring, (size_t)capacity);
    if (len < capacity)
      cstring[len] = '\0';
  }
  return (unsigned int)result.length();
}

std::vector<WordIndex> getWordIndices(AlignmentModel* alignmentModel, const char* sentence, bool source)
{
  std::vector<WordIndex> wordIndices;
  size_t i = 0;
  std::string s;
  while (sentence[i] != 0)
  {
    s = "";
    while (sentence[i] == ' ' && sentence[i] != 0)
    {
      ++i;
    }
    while (sentence[i] != ' ' && sentence[i] != 0)
    {
      s = s + sentence[i];
      ++i;
    }
    if (s != "")
    {
      WordIndex wordIndex = source ? alignmentModel->stringToSrcWordIndex(s) : alignmentModel->stringToTrgWordIndex(s);
      wordIndices.push_back(wordIndex);
    }
  }
  return wordIndices;
}

AlignmentModel* createAlignmentModel(int type, AlignmentModel* model = nullptr)
{
  switch ((AlignmentModelType)type)
  {
  case AlignmentModelType::Ibm1:
    return new Ibm1AlignmentModel();
  case AlignmentModelType::Ibm2:
    if (model != nullptr)
    {
      auto ibm1Model = dynamic_cast<Ibm1AlignmentModel*>(model);
      if (ibm1Model != nullptr)
        return new Ibm2AlignmentModel(*ibm1Model);
    }
    return new Ibm2AlignmentModel();
  case AlignmentModelType::Hmm:
    if (model != nullptr)
    {
      auto ibm1Model = dynamic_cast<Ibm1AlignmentModel*>(model);
      if (ibm1Model != nullptr)
        return new HmmAlignmentModel(*ibm1Model);
    }
    return new HmmAlignmentModel();
  case AlignmentModelType::Ibm3:
    if (model != nullptr)
    {
      auto hmmModel = dynamic_cast<HmmAlignmentModel*>(model);
      if (hmmModel != nullptr)
        return new Ibm3AlignmentModel(*hmmModel);
      auto ibm2Model = dynamic_cast<Ibm2AlignmentModel*>(model);
      if (ibm2Model != nullptr)
        return new Ibm3AlignmentModel(*ibm2Model);
    }
    return new Ibm3AlignmentModel();
  case AlignmentModelType::Ibm4:
    if (model != nullptr)
    {
      auto ibm3Model = dynamic_cast<Ibm3AlignmentModel*>(model);
      if (ibm3Model != nullptr)
        return new Ibm4AlignmentModel(*ibm3Model);
    }
    return new Ibm4AlignmentModel();
  case AlignmentModelType::IncrIbm1:
    return new IncrIbm1AlignmentModel();
  case AlignmentModelType::IncrIbm2:
    return new IncrIbm2AlignmentModel();
  case AlignmentModelType::IncrHmm:
    return new IncrHmmAlignmentModel();
  case AlignmentModelType::FastAlign:
    return new FastAlignModel();
  }
  return nullptr;
}

extern "C"
{
  void* smtModel_create(int alignmentModelType)
  {
    auto smtModelInfo = new SmtModelInfo;

    auto langModelInfo = new LangModelInfo;
    auto phrModelInfo = new PhraseModelInfo;
    auto swModelInfo = new SwModelInfo;

    phrModelInfo->phraseModelPars.ptsWeightVec.push_back(DEFAULT_PTS_WEIGHT);
    phrModelInfo->phraseModelPars.pstWeightVec.push_back(DEFAULT_PST_WEIGHT);

    langModelInfo->wpModel.reset(new WordPenaltyModel);
    langModelInfo->langModel.reset(new IncrJelMerNgramLM);

    phrModelInfo->invPhraseModel.reset(new WbaIncrPhraseModel);

    swModelInfo->swAligModels.push_back(std::shared_ptr<AlignmentModel>(createAlignmentModel(alignmentModelType)));
    swModelInfo->invSwAligModels.push_back(std::shared_ptr<AlignmentModel>(createAlignmentModel(alignmentModelType)));

    // Instantiate smt model
    smtModelInfo->smtModel.reset(new PhrLocalSwLiTm);

    // Link pointers
    smtModelInfo->smtModel->setLangModelInfo(langModelInfo);
    smtModelInfo->smtModel->setPhraseModelInfo(phrModelInfo);
    smtModelInfo->smtModel->setSwModelInfo(swModelInfo);
    smtModelInfo->smtModel->setTranslationMetadata(new TranslationMetadata<PhrScoreInfo>);

    return smtModelInfo;
  }

  bool smtModel_loadTranslationModel(void* smtModelHandle, const char* tmFileNamePrefix)
  {
    auto smtModelInfo = static_cast<SmtModelInfo*>(smtModelHandle);
    if (strcmp(smtModelInfo->tmFileNamePrefix.c_str(), tmFileNamePrefix) == 0)
      return true;

    smtModelInfo->tmFileNamePrefix = tmFileNamePrefix;
    return smtModelInfo->smtModel->loadAligModel(tmFileNamePrefix) == THOT_OK;
  }

  bool smtModel_loadLanguageModel(void* smtModelHandle, const char* lmFileName)
  {
    auto smtModelInfo = static_cast<SmtModelInfo*>(smtModelHandle);
    if (strcmp(smtModelInfo->lmFileName.c_str(), lmFileName) == 0)
      return true;

    smtModelInfo->lmFileName = lmFileName;
    return smtModelInfo->smtModel->loadLangModel(lmFileName) == THOT_OK;
  }

  void smtModel_setNonMonotonicity(void* smtModelHandle, unsigned int nomon)
  {
    auto smtModelInfo = static_cast<SmtModelInfo*>(smtModelHandle);
    smtModelInfo->smtModel->set_U_par(nomon);
  }

  void smtModel_setW(void* smtModelHandle, float w)
  {
    auto smtModelInfo = static_cast<SmtModelInfo*>(smtModelHandle);
    smtModelInfo->smtModel->set_W_par(w);
  }

  void smtModel_setA(void* smtModelHandle, unsigned int a)
  {
    auto smtModelInfo = static_cast<SmtModelInfo*>(smtModelHandle);
    smtModelInfo->smtModel->set_A_par(a);
  }

  void smtModel_setE(void* smtModelHandle, unsigned int e)
  {
    auto smtModelInfo = static_cast<SmtModelInfo*>(smtModelHandle);
    smtModelInfo->smtModel->set_E_par(e);
  }

  void smtModel_setHeuristic(void* smtModelHandle, unsigned int heuristic)
  {
    auto smtModelInfo = static_cast<SmtModelInfo*>(smtModelHandle);
    smtModelInfo->smtModel->setHeuristic(heuristic);
  }

  void smtModel_setOnlineTrainingParameters(void* smtModelHandle, unsigned int algorithm,
                                            unsigned int learningRatePolicy, float learnStepSize, unsigned int emIters,
                                            unsigned int e, unsigned int r)
  {
    auto smtModelInfo = static_cast<SmtModelInfo*>(smtModelHandle);
    OnlineTrainingPars otPars;
    otPars.onlineLearningAlgorithm = algorithm;
    otPars.learningRatePolicy = learningRatePolicy;
    otPars.learnStepSize = learnStepSize;
    otPars.emIters = emIters;
    otPars.E_par = e;
    otPars.R_par = r;
    smtModelInfo->smtModel->setOnlineTrainingPars(otPars);
  }

  void smtModel_setWeights(void* smtModelHandle, const float* weights, unsigned int capacity)
  {
    auto smtModelInfo = static_cast<SmtModelInfo*>(smtModelHandle);
    std::vector<float> weightsVec;
    for (unsigned int i = 0; i < capacity; ++i)
      weightsVec.push_back(weights[i]);
    smtModelInfo->smtModel->setWeights(weightsVec);
  }

  void* smtModel_getSingleWordAlignmentModel(void* smtModelHandle)
  {
    auto smtModelInfo = static_cast<SmtModelInfo*>(smtModelHandle);
    return smtModelInfo->smtModel->getSwModelInfo()->swAligModels[0].get();
  }

  void* smtModel_getInverseSingleWordAlignmentModel(void* smtModelHandle)
  {
    auto smtModelInfo = static_cast<SmtModelInfo*>(smtModelHandle);
    return smtModelInfo->smtModel->getSwModelInfo()->invSwAligModels[0].get();
  }

  bool smtModel_saveModels(void* smtModelHandle)
  {
    auto smtModelInfo = static_cast<SmtModelInfo*>(smtModelHandle);
    if (smtModelInfo->smtModel->printAligModel(smtModelInfo->tmFileNamePrefix) == THOT_ERROR)
      return false;

    return smtModelInfo->smtModel->printLangModel(smtModelInfo->lmFileName);
  }

  void smtModel_close(void* smtModelHandle)
  {
    auto smtModelInfo = static_cast<SmtModelInfo*>(smtModelHandle);
    smtModelInfo->smtModel->clear();
    delete smtModelInfo;
  }

  void* decoder_create(void* smtModelHandle)
  {
    auto smtModelInfo = static_cast<SmtModelInfo*>(smtModelHandle);

    auto stackDecoder = new multi_stack_decoder_rec<PhrLocalSwLiTm>;

    stackDecoder->setParentSmtModel(smtModelInfo->smtModel.get());
    // Create statistical machine translation model instance (it is
    // cloned from the main one)
    auto smtModel = dynamic_cast<PhrLocalSwLiTm*>(smtModelInfo->smtModel->clone());
    smtModel->setTranslationMetadata(new TranslationMetadata<PhrScoreInfo>);
    stackDecoder->setSmtModel(smtModel);

    stackDecoder->useBestScorePruning(true);

    return stackDecoder;
  }

  void decoder_setS(void* decoderHandle, unsigned int s)
  {
    auto stackDecoder = static_cast<multi_stack_decoder_rec<PhrLocalSwLiTm>*>(decoderHandle);
    stackDecoder->set_S_par(s);
  }

  void decoder_setBreadthFirst(void* decoderHandle, bool breadthFirst)
  {
    auto stackDecoder = static_cast<multi_stack_decoder_rec<PhrLocalSwLiTm>*>(decoderHandle);
    stackDecoder->set_breadthFirst(breadthFirst);
  }

  void decoder_setG(void* decoderHandle, unsigned int g)
  {
    auto stackDecoder = static_cast<multi_stack_decoder_rec<PhrLocalSwLiTm>*>(decoderHandle);
    stackDecoder->set_G_par(g);
  }

  void decoder_close(void* decoderHandle)
  {
    auto stackDecoder = static_cast<multi_stack_decoder_rec<PhrLocalSwLiTm>*>(decoderHandle);
    delete stackDecoder;
  }

  void* decoder_translate(void* decoderHandle, const char* sentence)
  {
    auto stackDecoder = static_cast<multi_stack_decoder_rec<PhrLocalSwLiTm>*>(decoderHandle);

    auto result = new TranslationData;

    // Use translator
    PhrLocalSwLiTm::Hypothesis hyp = stackDecoder->translate(sentence);

    std::vector<std::pair<PositionIndex, PositionIndex>> amatrix;
    // Obtain phrase alignment
    stackDecoder->getSmtModel()->aligMatrix(hyp, amatrix);
    stackDecoder->getSmtModel()->getPhraseAlignment(amatrix, result->sourceSegmentation, result->targetSegmentCuts);
    result->target = stackDecoder->getSmtModel()->getTransInPlainTextVec(hyp, result->targetUnknownWords);
    result->score = stackDecoder->getSmtModel()->getScoreForHyp(hyp);
    result->scoreComponents = stackDecoder->getSmtModel()->scoreCompsForHyp(hyp);

    return result;
  }

  unsigned int decoder_translateNBest(void* decoderHandle, unsigned int n, const char* sentence, void** results)
  {
    auto stackDecoder = static_cast<multi_stack_decoder_rec<PhrLocalSwLiTm>*>(decoderHandle);

    // Enable word graph generation
    stackDecoder->enableWordGraph();

    // Use translator
    stackDecoder->translate(sentence);
    WordGraph* wg = stackDecoder->getWordGraphPtr();

    stackDecoder->disableWordGraph();

    std::vector<TranslationData> translations;
    wg->obtainNbestList(n, translations);

    for (unsigned int i = 0; i < n && i < translations.size(); ++i)
      results[i] = new TranslationData(translations[i]);

    return (unsigned int)translations.size();
  }

  void* decoder_getWordGraph(void* decoderHandle, const char* sentence)
  {
    auto stackDecoder = static_cast<multi_stack_decoder_rec<PhrLocalSwLiTm>*>(decoderHandle);

    auto result = new WordGraphInfo;

    stackDecoder->useBestScorePruning(false);

    // Enable word graph generation
    stackDecoder->enableWordGraph();

    // Use translator
    PhrLocalSwLiTm::Hypothesis hyp = stackDecoder->translate(sentence);
    WordGraph* wg = stackDecoder->getWordGraphPtr();

    stackDecoder->disableWordGraph();

    stackDecoder->useBestScorePruning(true);

    if (stackDecoder->getSmtModel()->isComplete(hyp))
    {
      // Remove non-useful states from word-graph
      wg->obtainWgComposedOfUsefulStates();
      wg->orderArcsTopol();

      std::ostringstream outS;
      wg->print(outS, false);
      result->wordGraphStr = outS.str();
      result->initialStateScore = wg->getInitialStateScore();
    }
    else
    {
      result->wordGraphStr = "";
      result->initialStateScore = 0;
    }

    return result;
  }

  void* decoder_getBestPhraseAlignment(void* decoderHandle, const char* sentence, const char* translation)
  {
    auto stackDecoder = static_cast<multi_stack_decoder_rec<PhrLocalSwLiTm>*>(decoderHandle);

    auto result = new TranslationData();
    PhrLocalSwLiTm::Hypothesis hyp = stackDecoder->translateWithRef(sentence, translation);

    std::vector<std::pair<PositionIndex, PositionIndex>> amatrix;
    // Obtain phrase alignment
    stackDecoder->getSmtModel()->aligMatrix(hyp, amatrix);
    stackDecoder->getSmtModel()->getPhraseAlignment(amatrix, result->sourceSegmentation, result->targetSegmentCuts);
    result->target = stackDecoder->getSmtModel()->getTransInPlainTextVec(hyp, result->targetUnknownWords);
    result->score = stackDecoder->getSmtModel()->getScoreForHyp(hyp);
    result->scoreComponents = stackDecoder->getSmtModel()->scoreCompsForHyp(hyp);

    return result;
  }

  bool decoder_trainSentencePair(void* decoderHandle, const char* sourceSentence, const char* targetSentence)
  {
    auto stackDecoder = static_cast<multi_stack_decoder_rec<PhrLocalSwLiTm>*>(decoderHandle);

    // Obtain system translation
#ifdef THOT_ENABLE_UPDATE_LLWEIGHTS
    stackDecoder->enableWordGraph();
#endif

    PhrLocalSwLiTm::Hypothesis hyp = stackDecoder->translate(sourceSentence);
    std::string sysSent = stackDecoder->getSmtModel()->getTransInPlainText(hyp);

    // Add sentence to word-predictor
    stackDecoder->getParentSmtModel()->addSentenceToWordPred(StrProcUtils::stringToStringVector(targetSentence));

#ifdef THOT_ENABLE_UPDATE_LLWEIGHTS
    // Train log-linear weights

    // Retrieve pointer to wordgraph
    WordGraph* wgPtr = stackDecoder->getWordGraphPtr();
    stackDecoder->getParentSmtModel()->updateLogLinearWeights(targetSentence, wgPtr);

    stackDecoder->disableWordGraph();
#endif

    // Train generative models
    return stackDecoder->getParentSmtModel()->onlineTrainFeatsSentPair(sourceSentence, targetSentence, sysSent.c_str());
  }

  unsigned int tdata_getTarget(void* dataHandle, char* target, unsigned int capacity)
  {
    auto data = static_cast<TranslationData*>(dataHandle);
    return copyString(StrProcUtils::stringVectorToString(data->target), target, capacity);
  }

  unsigned int tdata_getPhraseCount(void* dataHandle)
  {
    auto data = static_cast<TranslationData*>(dataHandle);
    return (unsigned int)data->sourceSegmentation.size();
  }

  unsigned int tdata_getSourceSegmentation(void* dataHandle, unsigned int** sourceSegmentation, unsigned int capacity)
  {
    auto data = static_cast<TranslationData*>(dataHandle);
    if (sourceSegmentation != NULL)
    {
      for (unsigned int i = 0; i < capacity && i < data->sourceSegmentation.size(); i++)
      {
        sourceSegmentation[i][0] = data->sourceSegmentation[i].first;
        sourceSegmentation[i][1] = data->sourceSegmentation[i].second;
      }
    }
    return (unsigned int)data->sourceSegmentation.size();
  }

  unsigned int tdata_getTargetSegmentCuts(void* dataHandle, unsigned int* targetSegmentCuts, unsigned int capacity)
  {
    auto data = static_cast<TranslationData*>(dataHandle);
    if (targetSegmentCuts != NULL)
    {
      for (unsigned int i = 0; i < capacity && i < data->targetSegmentCuts.size(); i++)
        targetSegmentCuts[i] = data->targetSegmentCuts[i];
    }
    return (unsigned int)data->targetSegmentCuts.size();
  }

  unsigned int tdata_getTargetUnknownWords(void* dataHandle, unsigned int* targetUnknownWords, unsigned int capacity)
  {
    auto data = static_cast<TranslationData*>(dataHandle);
    if (targetUnknownWords != NULL)
    {
      unsigned int i = 0;
      for (std::set<PositionIndex>::const_iterator it = data->targetUnknownWords.begin();
           it != data->targetUnknownWords.end() && i < capacity; ++it)
      {
        targetUnknownWords[i] = *it;
        i++;
      }
    }
    return (unsigned int)data->targetUnknownWords.size();
  }

  double tdata_getScore(void* dataHandle)
  {
    auto data = static_cast<TranslationData*>(dataHandle);
    return data->score;
  }

  unsigned int tdata_getScoreComponents(void* dataHandle, double* scoreComps, unsigned int capacity)
  {
    auto data = static_cast<TranslationData*>(dataHandle);
    for (unsigned int i = 0; i < capacity && i < data->scoreComponents.size(); i++)
      scoreComps[i] = data->scoreComponents[i];
    return (unsigned int)data->scoreComponents.size();
  }

  void tdata_destroy(void* dataHandle)
  {
    auto data = static_cast<TranslationData*>(dataHandle);
    delete data;
  }

  unsigned int wg_getString(void* wgHandle, char* wordGraphStr, unsigned int capacity)
  {
    auto wordGraph = static_cast<WordGraphInfo*>(wgHandle);
    return copyString(wordGraph->wordGraphStr, wordGraphStr, capacity);
  }

  double wg_getInitialStateScore(void* wgHandle)
  {
    auto wg = static_cast<WordGraphInfo*>(wgHandle);
    return wg->initialStateScore;
  }

  void wg_destroy(void* wgHandle)
  {
    WordGraphInfo* wordGraph = static_cast<WordGraphInfo*>(wgHandle);
    delete wordGraph;
  }

  void* swAlignModel_create(int type, void* swAlignModelHandle)
  {
    return createAlignmentModel(type, static_cast<AlignmentModel*>(swAlignModelHandle));
  }

  void* swAlignModel_open(int type, const char* prefFileName)
  {
    AlignmentModel* alignmentModel = createAlignmentModel(type);
    if (alignmentModel->load(prefFileName) == THOT_ERROR)
    {
      delete alignmentModel;
      return NULL;
    }
    return alignmentModel;
  }

  unsigned int swAlignModel_getMaxSentenceLength(void* swAlignModelHandle)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    return alignmentModel->getMaxSentenceLength();
  }

  void swAlignModel_setVariationalBayes(void* swAlignModelHandle, bool variationalBayes)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    alignmentModel->setVariationalBayes(variationalBayes);
  }

  bool swAlignModel_getVariationalBayes(void* swAlignModelHandle)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    return alignmentModel->getVariationalBayes();
  }

  void swAlignModel_setFastAlignP0(void* swAlignModelHandle, double p0)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto fastAlignModel = dynamic_cast<FastAlignModel*>(alignmentModel);
    if (fastAlignModel != nullptr)
      fastAlignModel->setFastAlignP0(p0);
  }

  double swAlignModel_getFastAlignP0(void* swAlignModelHandle)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto fastAlignModel = dynamic_cast<FastAlignModel*>(alignmentModel);
    if (fastAlignModel != nullptr)
      return fastAlignModel->getFastAlignP0();
    return 0;
  }

  void swAlignModel_setHmmP0(void* swAlignModelHandle, double p0)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto hmmAlignmentModel = dynamic_cast<HmmAlignmentModel*>(alignmentModel);
    if (hmmAlignmentModel != nullptr)
      hmmAlignmentModel->setHmmP0(p0);
  }

  double swAlignModel_getHmmP0(void* swAlignModelHandle)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto hmmAlignmentModel = dynamic_cast<HmmAlignmentModel*>(alignmentModel);
    if (hmmAlignmentModel != nullptr)
      return hmmAlignmentModel->getHmmP0();
    return 0;
  }

  void swAlignModel_setHmmLexicalSmoothingFactor(void* swAlignModelHandle, double lexicalSmoothingFactor)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto hmmAlignmentModel = dynamic_cast<HmmAlignmentModel*>(alignmentModel);
    if (hmmAlignmentModel != nullptr)
      hmmAlignmentModel->setLexicalSmoothFactor(lexicalSmoothingFactor);
  }

  double swAlignModel_getHmmLexicalSmoothingFactor(void* swAlignModelHandle)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto hmmAlignmentModel = dynamic_cast<HmmAlignmentModel*>(alignmentModel);
    if (hmmAlignmentModel != nullptr)
      return hmmAlignmentModel->getLexicalSmoothFactor();
    return 0;
  }

  void swAlignModel_setHmmAlignmentSmoothingFactor(void* swAlignModelHandle, double alignmentSmoothingFactor)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto hmmAlignmentModel = dynamic_cast<HmmAlignmentModel*>(alignmentModel);
    if (hmmAlignmentModel != nullptr)
      hmmAlignmentModel->setHmmAlignmentSmoothFactor(alignmentSmoothingFactor);
  }

  double swAlignModel_getHmmAlignmentSmoothingFactor(void* swAlignModelHandle)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto hmmAlignmentModel = dynamic_cast<HmmAlignmentModel*>(alignmentModel);
    if (hmmAlignmentModel != nullptr)
      return hmmAlignmentModel->getHmmAlignmentSmoothFactor();
    return 0;
  }

  void swAlignModel_setIbm2CompactAlignmentTable(void* swAlignModelHandle, bool compactAlignmentTable)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto ibm2AlignmentModel = dynamic_cast<Ibm2AlignmentModel*>(alignmentModel);
    if (ibm2AlignmentModel != nullptr)
      ibm2AlignmentModel->setCompactAlignmentTable(compactAlignmentTable);
  }

  bool swAlignModel_getIbm2CompactAlignmentTable(void* swAlignModelHandle)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto ibm2AlignmentModel = dynamic_cast<Ibm2AlignmentModel*>(alignmentModel);
    if (ibm2AlignmentModel != nullptr)
      return ibm2AlignmentModel->getCompactAlignmentTable();
    return false;
  }

  void swAlignModel_setIbm3FertilitySmoothingFactor(void* swAlignModelHandle, double fertilitySmoothingFactor)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto ibm3AlignmentModel = dynamic_cast<Ibm3AlignmentModel*>(alignmentModel);
    if (ibm3AlignmentModel != nullptr)
      ibm3AlignmentModel->setFertilitySmoothFactor(fertilitySmoothingFactor);
  }

  double swAlignModel_getIbm3FertilitySmoothingFactor(void* swAlignModelHandle)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto ibm3AlignmentModel = dynamic_cast<Ibm3AlignmentModel*>(alignmentModel);
    if (ibm3AlignmentModel != nullptr)
      return ibm3AlignmentModel->getFertilitySmoothFactor();
    return 0;
  }

  void swAlignModel_setIbm3CountThreshold(void* swAlignModelHandle, double countThreshold)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto ibm3AlignmentModel = dynamic_cast<Ibm3AlignmentModel*>(alignmentModel);
    if (ibm3AlignmentModel != nullptr)
      ibm3AlignmentModel->setCountThreshold(countThreshold);
  }

  double swAlignModel_getIbm3CountThreshold(void* swAlignModelHandle)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto ibm3AlignmentModel = dynamic_cast<Ibm3AlignmentModel*>(alignmentModel);
    if (ibm3AlignmentModel != nullptr)
      return ibm3AlignmentModel->getCountThreshold();
    return 0;
  }

  void swAlignModel_setIbm4DistortionSmoothingFactor(void* swAlignModelHandle, double distortionSmoothingFactor)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto ibm4AlignmentModel = dynamic_cast<Ibm4AlignmentModel*>(alignmentModel);
    if (ibm4AlignmentModel != nullptr)
      ibm4AlignmentModel->setDistortionSmoothFactor(distortionSmoothingFactor);
  }

  double swAlignModel_getIbm4DistortionSmoothingFactor(void* swAlignModelHandle)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto ibm4AlignmentModel = dynamic_cast<Ibm4AlignmentModel*>(alignmentModel);
    if (ibm4AlignmentModel != nullptr)
      return ibm4AlignmentModel->getDistortionSmoothFactor();
    return 0;
  }

  unsigned int swAlignModel_getSourceWordCount(void* swAlignModelHandle)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    return (unsigned int)alignmentModel->getSrcVocabSize();
  }

  unsigned int swAlignModel_getSourceWord(void* swAlignModelHandle, unsigned int index, char* wordStr,
                                          unsigned int capacity)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    return copyString(alignmentModel->wordIndexToSrcString(index), wordStr, capacity);
  }

  unsigned int swAlignModel_getSourceWordIndex(void* swAlignModelHandle, const char* word)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    return alignmentModel->stringToSrcWordIndex(word);
  }

  unsigned int swAlignModel_getTargetWordCount(void* swAlignModelHandle)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    return (unsigned int)alignmentModel->getTrgVocabSize();
  }

  unsigned int swAlignModel_getTargetWord(void* swAlignModelHandle, unsigned int index, char* wordStr,
                                          unsigned int capacity)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    return copyString(alignmentModel->wordIndexToTrgString(index), wordStr, capacity);
  }

  unsigned int swAlignModel_getTargetWordIndex(void* swAlignModelHandle, const char* word)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    return alignmentModel->stringToTrgWordIndex(word);
  }

  void swAlignModel_addSentencePair(void* swAlignModelHandle, const char* sourceSentence, const char* targetSentence)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);

    std::vector<std::string> source = StrProcUtils::stringToStringVector(sourceSentence);
    std::vector<std::string> target = StrProcUtils::stringToStringVector(targetSentence);

    alignmentModel->addSentencePair(source, target, 1);
    for (unsigned int j = 0; j < source.size(); j++)
      alignmentModel->addSrcSymbol(source[j]);
    for (unsigned int j = 0; j < target.size(); j++)
      alignmentModel->addTrgSymbol(target[j]);
  }

  void swAlignModel_readSentencePairs(void* swAlignModelHandle, const char* sourceFilename, const char* targetFilename,
                                      const char* countsFilename)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    std::pair<unsigned int, unsigned int> sentRange;
    alignmentModel->readSentencePairs(sourceFilename, targetFilename, countsFilename, sentRange);
  }

  void swAlignModel_mapSourceWordToWordClass(void* swAlignModelHandle, const char* word, const char* wordClass)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto ibm4AlignmentModel = dynamic_cast<Ibm4AlignmentModel*>(alignmentModel);
    if (ibm4AlignmentModel != nullptr)
      ibm4AlignmentModel->mapSrcWordToWordClass(ibm4AlignmentModel->addSrcSymbol(word), wordClass);
  }

  void swAlignModel_mapTargetWordToWordClass(void* swAlignModelHandle, const char* word, const char* wordClass)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto ibm4AlignmentModel = dynamic_cast<Ibm4AlignmentModel*>(alignmentModel);
    if (ibm4AlignmentModel != nullptr)
      ibm4AlignmentModel->mapTrgWordToWordClass(ibm4AlignmentModel->addTrgSymbol(word), wordClass);
  }

  unsigned int swAlignModel_startTraining(void* swAlignModelHandle)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    return alignmentModel->startTraining();
  }

  void swAlignModel_train(void* swAlignModelHandle, unsigned int numIters)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    for (unsigned int i = 0; i < numIters; i++)
      alignmentModel->train();
  }

  void swAlignModel_endTraining(void* swAlignModelHandle)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    alignmentModel->endTraining();
  }

  void swAlignModel_save(void* swAlignModelHandle, const char* prefFileName)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    alignmentModel->print(prefFileName);
  }

  double swAlignModel_getTranslationProbability(void* swAlignModelHandle, const char* srcWord, const char* trgWord)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    WordIndex srcWordIndex = alignmentModel->stringToSrcWordIndex(srcWord);
    WordIndex trgWordIndex = alignmentModel->stringToTrgWordIndex(trgWord);
    return alignmentModel->translationProb(srcWordIndex, trgWordIndex);
  }

  double swAlignModel_getTranslationProbabilityByIndex(void* swAlignModelHandle, unsigned int srcWordIndex,
                                                       unsigned int trgWordIndex)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    return alignmentModel->translationProb(srcWordIndex, trgWordIndex);
  }

  double swAlignModel_getIbm2AlignmentProbability(void* swAlignModelHandle, unsigned int j, unsigned int sLen,
                                                  unsigned int tLen, unsigned int i)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto ibm2AlignmentModel = dynamic_cast<Ibm2AlignmentModel*>(alignmentModel);
    if (ibm2AlignmentModel != nullptr)
      return ibm2AlignmentModel->alignmentProb(j, sLen, tLen, i);
    auto faAlignmentModel = dynamic_cast<FastAlignModel*>(alignmentModel);
    if (faAlignmentModel != nullptr)
      return faAlignmentModel->alignmentProb(j, sLen, tLen, i);
    return 0;
  }

  double swAlignModel_getHmmAlignmentProbability(void* swAlignModelHandle, unsigned int prevI, unsigned int sLen,
                                                 unsigned int i)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto hmmAlignmentModel = dynamic_cast<HmmAlignmentModel*>(alignmentModel);
    if (hmmAlignmentModel != nullptr)
      return hmmAlignmentModel->hmmAlignmentProb(prevI, sLen, i);
    return 0;
  }

  double swAlignModel_getBestAlignment(void* swAlignModelHandle, const char* sourceSentence, const char* targetSentence,
                                       bool** matrix, unsigned int* iLen, unsigned int* jLen)
  {
    auto alignmentModel = static_cast<AlignmentModel*>(swAlignModelHandle);

    std::vector<WordIndex> sourceWordIndices = getWordIndices(alignmentModel, sourceSentence, true);
    std::vector<WordIndex> targetWordIndices = getWordIndices(alignmentModel, targetSentence, false);

    WordAlignmentMatrix waMatrix;
    LgProb prob = alignmentModel->getBestAlignment(sourceWordIndices, targetWordIndices, waMatrix);
    for (unsigned int i = 0; i < *iLen; i++)
    {
      for (unsigned int j = 0; j < *jLen; j++)
        matrix[i][j] = waMatrix.getValue(i, j);
    }
    *iLen = waMatrix.get_I();
    *jLen = waMatrix.get_J();
    return prob;
  }

  void* swAlignModel_getTranslations(void* swAlignModelHandle, const char* srcWord, double threshold)
  {
    auto swAligModelPtr = static_cast<AlignmentModel*>(swAlignModelHandle);
    WordIndex srcWordIndex = swAligModelPtr->stringToSrcWordIndex(srcWord);
    auto targetWordsPtr = new NbestTableNode<WordIndex>;
    if (swAligModelPtr->getEntriesForSource(srcWordIndex, *targetWordsPtr) && threshold > 0)
      targetWordsPtr->pruneGivenThreshold(threshold);
    return targetWordsPtr;
  }

  void* swAlignModel_getTranslationsByIndex(void* swAlignModelHandle, unsigned int srcWordIndex, double threshold)
  {
    auto swAligModelPtr = static_cast<AlignmentModel*>(swAlignModelHandle);
    auto targetWordsPtr = new NbestTableNode<WordIndex>;
    if (swAligModelPtr->getEntriesForSource(srcWordIndex, *targetWordsPtr) && threshold > 0)
      targetWordsPtr->pruneGivenThreshold(threshold);
    return targetWordsPtr;
  }

  void swAlignModel_close(void* swAlignModelHandle)
  {
    auto swAligModelPtr = static_cast<AlignmentModel*>(swAlignModelHandle);
    delete swAligModelPtr;
  }

  unsigned int swAlignTrans_getCount(void* swAlignTransHandle)
  {
    auto targetWordsPtr = static_cast<NbestTableNode<WordIndex>*>(swAlignTransHandle);
    return (unsigned int)targetWordsPtr->size();
  }

  unsigned int swAlignTrans_getTranslations(void* swAlignTransHandle, unsigned int* wordIndices, double* probs,
                                            unsigned int capacity)
  {
    auto targetWordsPtr = static_cast<NbestTableNode<WordIndex>*>(swAlignTransHandle);
    if (wordIndices != NULL || probs != NULL)
    {
      NbestTableNode<WordIndex>::iterator iter = targetWordsPtr->begin();
      for (unsigned int i = 0; i < capacity && iter != targetWordsPtr->end(); i++, iter++)
      {
        if (wordIndices != NULL)
          wordIndices[i] = iter->second;
        if (probs != NULL)
          probs[i] = iter->first;
      }
    }
    return (unsigned int)targetWordsPtr->size();
  }

  void swAlignTrans_destroy(void* swAlignTransHandle)
  {
    auto targetWordsPtr = static_cast<NbestTableNode<WordIndex>*>(swAlignTransHandle);
    delete targetWordsPtr;
  }

  bool giza_symmetr1(const char* lhsFileName, const char* rhsFileName, const char* outputFileName, bool transpose)
  {
    AlignmentExtractor alExt;
    if (alExt.open(lhsFileName) == THOT_ERROR)
      return false;
    alExt.symmetr1(rhsFileName, outputFileName, transpose);
    return true;
  }

  bool phraseModel_generate(const char* alignmentFileName, int maxPhraseLength, const char* tableFileName, int n)
  {
    _wbaIncrPhraseModel* phraseModelPtr = new WbaIncrPhraseModel;
    PhraseExtractParameters phePars;
    phePars.maxTrgPhraseLength = maxPhraseLength;
    bool result = phraseModelPtr->generateWbaIncrPhraseModel(alignmentFileName, phePars, false);
    if (result == THOT_OK)
      phraseModelPtr->printPhraseTable(tableFileName, n);
    delete phraseModelPtr;
    return result;
  }

  void* langModel_open(const char* prefFileName)
  {
    BaseNgramLM<LM_State>* lmPtr = new IncrJelMerNgramLM;
    if (lmPtr->load(prefFileName) == THOT_ERROR)
    {
      delete lmPtr;
      return NULL;
    }
    return lmPtr;
  }

  double langModel_getSentenceProbability(void* lmHandle, const char* sentence)
  {
    BaseNgramLM<LM_State>* lmPtr = static_cast<BaseNgramLM<LM_State>*>(lmHandle);
    return lmPtr->getSentenceLog10ProbStr(StrProcUtils::stringToStringVector(sentence));
  }

  void langModel_close(void* lmHandle)
  {
    BaseNgramLM<LM_State>* lmPtr = static_cast<BaseNgramLM<LM_State>*>(lmHandle);
    delete lmPtr;
  }

  void* llWeightUpdater_create()
  {
    auto llWeightUpdater = new KbMiraLlWu;
    llWeightUpdater->setScorer(new MiraBleu);
    return llWeightUpdater;
  }

  void llWeightUpdater_updateClosedCorpus(void* llWeightUpdaterHandle, const char** references, const char*** nblists,
                                          const double*** scoreComps, const unsigned int* nblistLens, float* weights,
                                          unsigned int numSents, unsigned int numWeights)
  {
    auto llWeightUpdater = static_cast<KbMiraLlWu*>(llWeightUpdaterHandle);

    std::vector<std::string> refsVec;
    std::vector<std::vector<std::string>> nblistsVec;
    std::vector<std::vector<std::vector<double>>> scoreCompsVec;
    for (unsigned int i = 0; i < numSents; ++i)
    {
      refsVec.push_back(references[i]);
      std::vector<std::string> nblistVec;
      std::vector<std::vector<double>> nblistScoreCompsVec;
      for (unsigned int j = 0; j < nblistLens[i]; ++j)
      {
        nblistVec.push_back(nblists[i][j]);
        std::vector<double> transScoreCompsVec;
        for (unsigned int k = 0; k < numWeights; ++k)
          transScoreCompsVec.push_back(scoreComps[i][j][k]);
        nblistScoreCompsVec.push_back(transScoreCompsVec);
      }
      nblistsVec.push_back(nblistVec);
      scoreCompsVec.push_back(nblistScoreCompsVec);
    }

    std::vector<double> curWeightsVec;
    for (unsigned int i = 0; i < numWeights; ++i)
      curWeightsVec.push_back(weights[i]);

    std::vector<double> newWeightsVec;
    llWeightUpdater->updateClosedCorpus(refsVec, nblistsVec, scoreCompsVec, curWeightsVec, newWeightsVec);

    for (unsigned int i = 0; i < numWeights; ++i)
      weights[i] = (float)newWeightsVec[i];
  }

  void llWeightUpdater_close(void* llWeightUpdaterHandle)
  {
    auto llWeightUpdater = static_cast<KbMiraLlWu*>(llWeightUpdaterHandle);
    delete llWeightUpdater;
  }
}
