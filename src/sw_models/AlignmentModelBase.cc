#include "sw_models/AlignmentModelBase.h"

#include "nlp_common/ErrorDefs.h"
#include "nlp_common/StrProcUtils.h"

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

using namespace std;

AlignmentModelBase::AlignmentModelBase()
    : alpha{0.01}, variationalBayes{false}, swVocab{make_shared<SingleWordVocab>()},
      sentenceHandler{make_shared<LightSentenceHandler>()}, wordClasses{std::make_shared<WordClasses>()}
{
}

AlignmentModelBase::AlignmentModelBase(AlignmentModelBase& model)
    : alpha{model.alpha}, variationalBayes{model.variationalBayes}, swVocab{model.swVocab},
      sentenceHandler{model.sentenceHandler}, wordClasses{model.wordClasses}
{
}

bool AlignmentModelBase::modelReadsAreProcessSafe()
{
  // By default it will be assumed that model reads are thread safe,
  // those unsafe classes will override this method returning false
  // instead
  return true;
}

void AlignmentModelBase::setVariationalBayes(bool variationalBayes)
{
  this->variationalBayes = variationalBayes;
}

bool AlignmentModelBase::getVariationalBayes()
{
  return variationalBayes;
}

bool AlignmentModelBase::readSentencePairs(const char* srcFileName, const char* trgFileName, const char* sentCountsFile,
                                           pair<unsigned int, unsigned int>& sentRange, int verbose)
{
  return sentenceHandler->readSentencePairs(srcFileName, trgFileName, sentCountsFile, sentRange, verbose);
}

pair<unsigned int, unsigned int> AlignmentModelBase::addSentencePair(vector<string> srcSentStr,
                                                                     vector<string> trgSentStr, Count c)
{
  return sentenceHandler->addSentencePair(srcSentStr, trgSentStr, c);
}

unsigned int AlignmentModelBase::numSentencePairs()
{
  return sentenceHandler->numSentencePairs();
}

int AlignmentModelBase::getSentencePair(unsigned int n, vector<string>& srcSentStr, vector<string>& trgSentStr,
                                        Count& c)
{
  return sentenceHandler->getSentencePair(n, srcSentStr, trgSentStr, c);
}

PositionIndex AlignmentModelBase::getMaxSentenceLength()
{
  return maxSentenceLength;
}

pair<double, double> AlignmentModelBase::loglikelihoodForAllSentences(int verbosity)
{
  pair<unsigned int, unsigned int> sentPairRange = make_pair(0, numSentencePairs() - 1);
  return loglikelihoodForPairRange(sentPairRange, verbosity);
}

LgProb AlignmentModelBase::computeLogProb(const char* srcSentence, const char* trgSentence,
                                          const WordAlignmentMatrix& aligMatrix, int verbose)
{
  vector<string> sSentVec, tSentVec;

  sSentVec = StrProcUtils::charItemsToVector(srcSentence);
  tSentVec = StrProcUtils::charItemsToVector(trgSentence);
  return computeLogProb(sSentVec, tSentVec, aligMatrix, verbose);
}

LgProb AlignmentModelBase::computeLogProb(const vector<string>& srcSentence, const vector<string>& trgSentence,
                                          const WordAlignmentMatrix& aligMatrix, int verbose)
{
  vector<WordIndex> sIndexVector = strVectorToSrcIndexVector(srcSentence);
  vector<WordIndex> tIndexVector = strVectorToTrgIndexVector(trgSentence);
  return computeLogProb(sIndexVector, tIndexVector, aligMatrix, verbose);
}

LgProb AlignmentModelBase::computeSumLogProb(const char* srcSentence, const char* trgSentence, int verbose)
{
  vector<string> sSentVec, tSentVec;

  sSentVec = StrProcUtils::charItemsToVector(srcSentence);
  tSentVec = StrProcUtils::charItemsToVector(trgSentence);
  return computeSumLogProb(sSentVec, tSentVec, verbose);
}

LgProb AlignmentModelBase::computeSumLogProb(const vector<string>& srcSentence, const vector<string>& trgSentence,
                                             int verbose)
{
  vector<WordIndex> sIndexVector, tIndexVector;

  sIndexVector = strVectorToSrcIndexVector(srcSentence);
  tIndexVector = strVectorToTrgIndexVector(trgSentence);

  return computeSumLogProb(sIndexVector, tIndexVector, verbose);
}

LgProb AlignmentModelBase::computePhraseSumLogProb(const vector<WordIndex>& srcPhrase,
                                                   const vector<WordIndex>& trgPhrase, int verbose)
{
  return computeSumLogProb(srcPhrase, trgPhrase, verbose);
}

bool AlignmentModelBase::getBestAlignments(const char* sourceTestFileName, const char* targetTestFilename,
                                           const char* outFileName)
{
  AwkInputStream srcTest, trgTest;
  WordAlignmentMatrix waMatrix;
  vector<PositionIndex> bestAlig;
  LgProb bestLgProb;
  ofstream outF;

  outF.open(outFileName, ios::out);
  if (!outF)
  {
    cerr << "Error while opening output file." << endl;
    return 1;
  }

  if (srcTest.open(sourceTestFileName) == THOT_ERROR)
  {
    cerr << "Error in source test file, file " << sourceTestFileName << " does not exist.\n";
    return THOT_ERROR;
  }
  if (trgTest.open(targetTestFilename) == THOT_ERROR)
  {
    cerr << "Error in target test file, file " << targetTestFilename << " does not exist.\n";
    return THOT_ERROR;
  }
  while (srcTest.getln())
  {
    if (trgTest.getln())
    {
      if (srcTest.NF > 0 && trgTest.NF > 0)
      {
        bestLgProb = getBestAlignment((char*)srcTest.dollar(0).c_str(), (char*)trgTest.dollar(0).c_str(), waMatrix);
        outF << "# Sentence pair " << srcTest.FNR << " ";
        waMatrix.getAligVec(bestAlig);
        printAligInGizaFormat((char*)srcTest.dollar(0).c_str(), (char*)trgTest.dollar(0).c_str(), bestLgProb.get_p(),
                              bestAlig, outF);
      }
    }
    else
    {
      cerr << "Error: Source and target test files have not the same size." << endl;
    }
  }
  outF.close();

  return THOT_OK;
}

LgProb AlignmentModelBase::getBestAlignment(const char* srcSentence, const char* trgSentence,
                                            WordAlignmentMatrix& bestWaMatrix)
{
  vector<string> sourceVector = StrProcUtils::charItemsToVector(srcSentence);
  vector<string> targetVector = StrProcUtils::charItemsToVector(trgSentence);
  return getBestAlignment(sourceVector, targetVector, bestWaMatrix);
}

LgProb AlignmentModelBase::getBestAlignment(const vector<string>& srcSentence, const vector<string>& trgSentence,
                                            WordAlignmentMatrix& bestWaMatrix)
{
  vector<WordIndex> srcWordIndexVector = strVectorToSrcIndexVector(srcSentence);
  vector<WordIndex> trgWordIndexVector = strVectorToTrgIndexVector(trgSentence);
  return getBestAlignment(srcWordIndexVector, trgWordIndexVector, bestWaMatrix);
}

LgProb AlignmentModelBase::getBestAlignment(const vector<WordIndex>& srcSentence, const vector<WordIndex>& trgSentence,
                                            WordAlignmentMatrix& bestWaMatrix)
{
  vector<PositionIndex> bestAlignment;
  LgProb logProb = getBestAlignment(srcSentence, trgSentence, bestAlignment);
  bestWaMatrix.init((PositionIndex)srcSentence.size(), (PositionIndex)trgSentence.size());
  bestWaMatrix.putAligVec(bestAlignment);
  return logProb;
}

LgProb AlignmentModelBase::getBestAlignment(const char* srcSentence, const char* trgSentence,
                                            vector<PositionIndex>& bestAlignment)
{
  vector<string> sourceVector = StrProcUtils::charItemsToVector(srcSentence);
  vector<string> targetVector = StrProcUtils::charItemsToVector(trgSentence);
  return getBestAlignment(sourceVector, targetVector, bestAlignment);
}

LgProb AlignmentModelBase::getBestAlignment(const vector<string>& srcSentence, const vector<string>& trgSentence,
                                            vector<PositionIndex>& bestAlignment)
{
  vector<WordIndex> srcWordIndexVector = strVectorToSrcIndexVector(srcSentence);
  vector<WordIndex> trgWordIndexVector = strVectorToTrgIndexVector(trgSentence);
  return getBestAlignment(srcWordIndexVector, trgWordIndexVector, bestAlignment);
}

ostream& AlignmentModelBase::printAligInGizaFormat(const char* sourceSentence, const char* targetSentence, Prob p,
                                                   vector<PositionIndex> alig, ostream& outS)
{
  vector<string> targetVector, sourceVector;
  unsigned int i, j;

  outS << "alignment score : " << p << endl;
  outS << targetSentence << endl;
  sourceVector = StrProcUtils::charItemsToVector(sourceSentence);
  targetVector = StrProcUtils::charItemsToVector(targetSentence);

  outS << "NULL ({ ";
  for (j = 0; j < alig.size(); ++j)
    if (alig[j] == 0)
      outS << j + 1 << " ";
  outS << "}) ";
  for (i = 0; i < sourceVector.size(); ++i)
  {
    outS << sourceVector[i] << " ({ ";
    for (j = 0; j < alig.size(); ++j)
      if (alig[j] == i + 1)
        outS << j + 1 << " ";
    outS << "}) ";
  }
  outS << endl;
  return outS;
}

bool AlignmentModelBase::loadGIZASrcVocab(const char* srcInputVocabFileName, int verbose)
{
  return swVocab->loadGIZASrcVocab(srcInputVocabFileName, verbose);
}

bool AlignmentModelBase::loadGIZATrgVocab(const char* trgInputVocabFileName, int verbose)
{
  return swVocab->loadGIZATrgVocab(trgInputVocabFileName, verbose);
}

bool AlignmentModelBase::printGIZASrcVocab(const char* srcOutputVocabFileName)
{
  return swVocab->printSrcVocab(srcOutputVocabFileName);
}

bool AlignmentModelBase::printGIZATrgVocab(const char* trgOutputVocabFileName)
{
  return swVocab->printTrgVocab(trgOutputVocabFileName);
}

bool AlignmentModelBase::printSentencePairs(const char* srcSentFile, const char* trgSentFile,
                                            const char* sentCountsFile)
{
  return sentenceHandler->printSentencePairs(srcSentFile, trgSentFile, sentCountsFile);
}

size_t AlignmentModelBase::getSrcVocabSize() const
{
  return swVocab->getSrcVocabSize();
}

WordIndex AlignmentModelBase::stringToSrcWordIndex(string s) const
{
  return swVocab->stringToSrcWordIndex(s);
}

string AlignmentModelBase::wordIndexToSrcString(WordIndex w) const
{
  return swVocab->wordIndexToSrcString(w);
}

bool AlignmentModelBase::existSrcSymbol(string s) const
{
  return swVocab->existSrcSymbol(s);
}

vector<WordIndex> AlignmentModelBase::strVectorToSrcIndexVector(vector<string> s)
{
  return swVocab->strVectorToSrcIndexVector(s);
}

WordIndex AlignmentModelBase::addSrcSymbol(string s)
{
  return swVocab->addSrcSymbol(s);
}

size_t AlignmentModelBase::getTrgVocabSize() const
{
  return swVocab->getTrgVocabSize();
}

WordIndex AlignmentModelBase::stringToTrgWordIndex(string t) const
{
  return swVocab->stringToTrgWordIndex(t);
}

string AlignmentModelBase::wordIndexToTrgString(WordIndex w) const
{
  return swVocab->wordIndexToTrgString(w);
}

bool AlignmentModelBase::existTrgSymbol(string t) const
{
  return swVocab->existTrgSymbol(t);
}

vector<WordIndex> AlignmentModelBase::strVectorToTrgIndexVector(vector<string> t)
{
  return swVocab->strVectorToTrgIndexVector(t);
}

WordIndex AlignmentModelBase::addTrgSymbol(string t)
{
  return swVocab->addTrgSymbol(t);
}

void AlignmentModelBase::clear()
{
  swVocab->clear();
  clearInfoAboutSentenceRange();
  clearTempVars();
  clearSentenceLengthModel();
  wordClasses->clear();
}

void AlignmentModelBase::clearInfoAboutSentenceRange()
{
  // Clear info about sentence range
  sentenceHandler->clear();
}

bool AlignmentModelBase::loadVariationalBayes(const string& filename)
{
  ifstream in(filename);
  if (!in)
    return THOT_OK;
  in >> variationalBayes >> alpha;

  return THOT_OK;
}

bool AlignmentModelBase::sentenceLengthIsOk(const std::vector<WordIndex> sentence)
{
  return !sentence.empty() && sentence.size() <= getMaxSentenceLength();
}

void AlignmentModelBase::loadConfig(const YAML::Node& config)
{
  variationalBayes = config["variationalBayes"].as<bool>();
  alpha = config["alpha"].as<double>();
}

bool AlignmentModelBase::loadOldConfig(const char* prefFileName, int verbose)
{
  string variationalBayesFile = prefFileName;
  variationalBayesFile = variationalBayesFile + ".var_bayes";
  return loadVariationalBayes(variationalBayesFile);
}

void AlignmentModelBase::createConfig(YAML::Emitter& out)
{
  out << YAML::Key << "model" << YAML::Value << getModelTypeStr();
  out << YAML::Key << "variationalBayes" << YAML::Value << variationalBayes;
  out << YAML::Key << "alpha" << YAML::Value << alpha;
}

vector<WordIndex> AlignmentModelBase::addNullWordToWidxVec(const vector<WordIndex>& vw)
{
  vector<WordIndex> result;

  result.push_back(NULL_WORD);
  for (unsigned int i = 0; i < vw.size(); ++i)
    result.push_back(vw[i]);

  return result;
}

vector<string> AlignmentModelBase::addNullWordToStrVec(const vector<string>& vw)
{
  vector<string> result;

  result.push_back(NULL_WORD_STR);
  for (unsigned int i = 0; i < vw.size(); ++i)
    result.push_back(vw[i]);

  return result;
}

WordClassIndex AlignmentModelBase::addSrcWordClass(const std::string& c)
{
  return wordClasses->addSrcWordClass(c);
}

WordClassIndex AlignmentModelBase::addTrgWordClass(const std::string& c)
{
  return wordClasses->addTrgWordClass(c);
}

void AlignmentModelBase::mapSrcWordToWordClass(WordIndex s, const std::string& c)
{
  wordClasses->mapSrcWordToWordClass(s, c);
}

void AlignmentModelBase::mapSrcWordToWordClass(WordIndex s, WordClassIndex c)
{
  wordClasses->mapSrcWordToWordClass(s, c);
}

void AlignmentModelBase::mapTrgWordToWordClass(WordIndex t, const std::string& c)
{
  wordClasses->mapTrgWordToWordClass(t, c);
}

void AlignmentModelBase::mapTrgWordToWordClass(WordIndex t, WordClassIndex c)
{
  wordClasses->mapTrgWordToWordClass(t, c);
}

bool AlignmentModelBase::load(const char* prefFileName, int verbose)
{
  if (prefFileName[0] != 0)
  {
    bool retVal;

    try
    {
      string configFileName = prefFileName;
      configFileName = configFileName + ".yml";
      YAML::Node config = YAML::LoadFile(configFileName);
      loadConfig(config);
    }
    catch (const YAML::BadFile&)
    {
      retVal = loadOldConfig(prefFileName, verbose);
      if (retVal == THOT_ERROR)
        return THOT_ERROR;
    }
    catch (const YAML::InvalidNode&)
    {
      retVal = loadOldConfig(prefFileName, verbose);
      if (retVal == THOT_ERROR)
        return THOT_ERROR;
    }

    // Load vocabularies if they exist
    string srcVocFileName = prefFileName;
    srcVocFileName = srcVocFileName + ".svcb";
    loadGIZASrcVocab(srcVocFileName.c_str(), verbose);

    string trgVocFileName = prefFileName;
    trgVocFileName = trgVocFileName + ".tvcb";
    loadGIZATrgVocab(trgVocFileName.c_str(), verbose);

    // Load files with source and target sentences
    // Warning: this must be made before reading file with anji
    // values
    string srcsFile = prefFileName;
    srcsFile = srcsFile + ".src";
    string trgsFile = prefFileName;
    trgsFile = trgsFile + ".trg";
    string srctrgcFile = prefFileName;
    srctrgcFile = srctrgcFile + ".srctrgc";
    pair<unsigned int, unsigned int> pui;
    retVal = readSentencePairs(srcsFile.c_str(), trgsFile.c_str(), srctrgcFile.c_str(), pui, verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    wordClasses->load(prefFileName, verbose);

    return THOT_OK;
  }
  else
    return THOT_ERROR;
}

bool AlignmentModelBase::print(const char* prefFileName, int verbose)
{
  bool retVal;

  YAML::Emitter out;
  out.SetDoublePrecision(std::numeric_limits<double>::digits10);
  out << YAML::BeginMap;
  createConfig(out);
  out << YAML::EndMap;

  string configFileName = prefFileName;
  configFileName = configFileName + ".yml";
  std::ofstream configFile(configFileName);
  configFile << out.c_str();
  configFile.close();

  // Print vocabularies
  string srcVocFileName = prefFileName;
  srcVocFileName = srcVocFileName + ".svcb";
  retVal = printGIZASrcVocab(srcVocFileName.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  string trgVocFileName = prefFileName;
  trgVocFileName = trgVocFileName + ".tvcb";
  retVal = printGIZATrgVocab(trgVocFileName.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print files with source and target sentences to temp files
  string srcsFileTemp = prefFileName;
  srcsFileTemp = srcsFileTemp + ".src.tmp";
  string trgsFileTemp = prefFileName;
  trgsFileTemp = trgsFileTemp + ".trg.tmp";
  string srctrgcFileTemp = prefFileName;
  srctrgcFileTemp = srctrgcFileTemp + ".srctrgc.tmp";
  retVal = printSentencePairs(srcsFileTemp.c_str(), trgsFileTemp.c_str(), srctrgcFileTemp.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // close sentence files
  sentenceHandler->clear();

  string srcsFile = prefFileName;
  srcsFile = srcsFile + ".src";
  string trgsFile = prefFileName;
  trgsFile = trgsFile + ".trg";
  string srctrgcFile = prefFileName;
  srctrgcFile = srctrgcFile + ".srctrgc";

  // move temp files to real destination
#ifdef _WIN32
  if (!MoveFileExA(srcsFileTemp.c_str(), srcsFile.c_str(), MOVEFILE_REPLACE_EXISTING))
    return THOT_ERROR;
  if (!MoveFileExA(trgsFileTemp.c_str(), trgsFile.c_str(), MOVEFILE_REPLACE_EXISTING))
    return THOT_ERROR;
  if (!MoveFileExA(srctrgcFileTemp.c_str(), srctrgcFile.c_str(), MOVEFILE_REPLACE_EXISTING))
    return THOT_ERROR;
#else
  if (rename(srcsFileTemp.c_str(), srcsFile.c_str()) != 0)
    return THOT_ERROR;
  if (rename(trgsFileTemp.c_str(), trgsFile.c_str()) != 0)
    return THOT_ERROR;
  if (rename(srctrgcFileTemp.c_str(), srctrgcFile.c_str()) != 0)
    return THOT_ERROR;
#endif

  // reload sentence files
  pair<unsigned int, unsigned int> pui;
  retVal = readSentencePairs(srcsFile.c_str(), trgsFile.c_str(), srctrgcFile.c_str(), pui, verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  retVal = wordClasses->print(prefFileName, verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  return THOT_OK;
}
