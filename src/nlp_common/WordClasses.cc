#include "nlp_common/WordClasses.h"

#include "nlp_common/ErrorDefs.h"
#include "nlp_common/StrProcUtils.h"

#include <fstream>
#include <iostream>

WordClasses::WordClasses()
{
  srcWordClassNames[NULL_WORD_CLASS_STR] = NULL_WORD_CLASS;
  trgWordClassNames[NULL_WORD_CLASS_STR] = NULL_WORD_CLASS;
}

WordClassIndex WordClasses::addSrcWordClass(const std::string& c)
{
  WordClassIndex wordClassIndex;
  auto iter = srcWordClassNames.find(c);
  if (iter == srcWordClassNames.end())
  {
    wordClassIndex = srcWordClassNames.size();
    srcWordClassNames[c] = wordClassIndex;
  }
  else
  {
    wordClassIndex = iter->second;
  }
  return wordClassIndex;
}

WordClassIndex WordClasses::addTrgWordClass(const std::string& c)
{
  WordClassIndex wordClassIndex;
  auto iter = trgWordClassNames.find(c);
  if (iter == trgWordClassNames.end())
  {
    wordClassIndex = trgWordClassNames.size();
    trgWordClassNames[c] = wordClassIndex;
  }
  else
  {
    wordClassIndex = iter->second;
  }
  return wordClassIndex;
}

WordClassIndex WordClasses::mapSrcWordToWordClass(WordIndex s, const std::string& c)
{
  WordClassIndex wordClassIndex = addSrcWordClass(c);
  mapSrcWordToWordClass(s, wordClassIndex);
  return wordClassIndex;
}

WordClassIndex WordClasses::mapTrgWordToWordClass(WordIndex t, const std::string& c)
{
  WordClassIndex wordClassIndex = addTrgWordClass(c);
  mapTrgWordToWordClass(t, wordClassIndex);
  return wordClassIndex;
}

void WordClasses::mapSrcWordToWordClass(WordIndex s, WordClassIndex c)
{
  if (srcWordClasses.size() <= s)
    srcWordClasses.resize(size_t{s} + 1);
  srcWordClasses[s] = c;
}

void WordClasses::mapTrgWordToWordClass(WordIndex t, WordClassIndex c)
{
  if (trgWordClasses.size() <= t)
    trgWordClasses.resize(size_t{t} + 1);
  trgWordClasses[t] = c;
}

WordClassIndex WordClasses::getSrcWordClass(WordIndex s) const
{
  if (s >= srcWordClasses.size())
    return NULL_WORD_CLASS;
  return srcWordClasses[s];
}

WordClassIndex WordClasses::getTrgWordClass(WordIndex t) const
{
  if (t >= trgWordClasses.size())
    return NULL_WORD_CLASS;
  return trgWordClasses[t];
}

WordClassIndex WordClasses::getSrcWordClassCount() const
{
  return srcWordClassNames.size();
}

WordClassIndex WordClasses::getTrgWordClassCount() const
{
  return trgWordClassNames.size();
}

bool WordClasses::load(const char* prefFileName, int verbose)
{
  // Load file with source word class names
  std::string srcWordClassNamesFile = prefFileName;
  srcWordClassNamesFile = srcWordClassNamesFile + ".src_class_names";
  bool retVal = loadSrcWordClassNames(srcWordClassNamesFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with source word classes
  std::string srcWordClassesFile = prefFileName;
  srcWordClassesFile = srcWordClassesFile + ".src_classes";
  retVal = loadSrcWordClasses(srcWordClassesFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with target word class names
  std::string trgWordClassNamesFile = prefFileName;
  trgWordClassNamesFile = trgWordClassNamesFile + ".trg_class_names";
  retVal = loadTrgWordClassNames(trgWordClassNamesFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with target word classes
  std::string trgWordClassesFile = prefFileName;
  trgWordClassesFile = trgWordClassesFile + ".trg_classes";
  retVal = loadTrgWordClasses(trgWordClassesFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  return THOT_OK;
}

bool WordClasses::print(const char* prefFileName, int verbose) const
{
  // Print file with source word class names
  std::string srcWordClassNamesFile = prefFileName;
  srcWordClassNamesFile = srcWordClassNamesFile + ".src_class_names";
  bool retVal = printSrcWordClassNames(srcWordClassNamesFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with source word classes
  std::string srcWordClassesFile = prefFileName;
  srcWordClassesFile = srcWordClassesFile + ".src_classes";
  retVal = printSrcWordClasses(srcWordClassesFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with target word class names
  std::string trgWordClassNamesFile = prefFileName;
  trgWordClassNamesFile = trgWordClassNamesFile + ".trg_class_names";
  retVal = printTrgWordClassNames(trgWordClassNamesFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with target word classes
  std::string trgWordClassesFile = prefFileName;
  trgWordClassesFile = trgWordClassesFile + ".trg_classes";
  retVal = printTrgWordClasses(trgWordClassesFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  return THOT_OK;
}

bool WordClasses::loadSrcWordClasses(const char* srcWordClassesFile, int verbose)
{
  return loadWordClasses(srcWordClassesFile, srcWordClasses, verbose);
}

bool WordClasses::loadTrgWordClasses(const char* trgWordClassesFile, int verbose)
{
  return loadWordClasses(trgWordClassesFile, trgWordClasses, verbose);
}

bool WordClasses::loadSrcWordClassNames(const char* srcWordClassNamesFile, int verbose)
{
  return loadWordClassNames(srcWordClassNamesFile, srcWordClassNames, verbose);
}

bool WordClasses::loadTrgWordClassNames(const char* trgWordClassNamesFile, int verbose)
{
  return loadWordClassNames(trgWordClassNamesFile, trgWordClassNames, verbose);
}

bool WordClasses::printSrcWordClasses(const char* srcWordClassesFile, int verbose) const
{
  return printWordClasses(srcWordClassesFile, srcWordClasses, verbose);
}

bool WordClasses::printTrgWordClasses(const char* trgWordClassesFile, int verbose) const
{
  return printWordClasses(trgWordClassesFile, trgWordClasses, verbose);
}

bool WordClasses::printSrcWordClassNames(const char* srcWordClassNamesFile, int verbose) const
{
  return printWordClassNames(srcWordClassNamesFile, srcWordClassNames, verbose);
}

bool WordClasses::printTrgWordClassNames(const char* trgWordClassNamesFile, int verbose) const
{
  return printWordClassNames(trgWordClassNamesFile, trgWordClassNames, verbose);
}

void WordClasses::clear()
{
  srcWordClasses.clear();
  trgWordClasses.clear();
  srcWordClassNames.clear();
  trgWordClassNames.clear();

  srcWordClassNames[NULL_WORD_CLASS_STR] = NULL_WORD_CLASS;
  trgWordClassNames[NULL_WORD_CLASS_STR] = NULL_WORD_CLASS;
}

bool WordClasses::loadWordClasses(const char* wordClassesFile, std::vector<WordClassIndex>& wordClasses, int verbose)
{
  wordClasses.clear();

  if (verbose)
    std::cerr << "Loading word classes from " << wordClassesFile << std::endl;

  // Try to open file
  std::ifstream inF(wordClassesFile, std::ios::in | std::ios::binary);
  if (!inF)
  {
    if (verbose)
      std::cerr << "Word classes file " << wordClassesFile << " does not exist." << std::endl;
    return THOT_ERROR;
  }
  else
  {
    // Read register
    bool end = false;
    while (!end)
    {
      WordIndex w;
      WordClassIndex c;
      if (inF.read((char*)&w, sizeof(WordIndex)))
      {
        inF.read((char*)&c, sizeof(WordClassIndex));

        if (wordClasses.size() <= w)
          wordClasses.resize(size_t{w} + 1);
        wordClasses[w] = c;
      }
      else
      {
        end = true;
      }
    }
    return THOT_OK;
  }
}

bool WordClasses::loadWordClassNames(const char* wordClassNamesFile,
                                     std::unordered_map<std::string, WordClassIndex>& wordClassNames, int verbose)
{
  wordClassNames.clear();

  if (verbose)
    std::cerr << "Loading word class names from " << wordClassNamesFile << std::endl;

  // Try to open file
  std::ifstream inF(wordClassNamesFile);
  if (!inF)
  {
    if (verbose)
      std::cerr << "Word class names file " << wordClassNamesFile << " does not exist." << std::endl;
    return THOT_ERROR;
  }
  else
  {
    std::string line;
    while (std::getline(inF, line))
    {
      std::vector<std::string> parts = StrProcUtils::split(line, '\t');
      WordClassIndex wordClassIndex = std::stoi(parts[1]);
      wordClassNames[parts[0]] = wordClassIndex;
    }
    return THOT_OK;
  }
}

bool WordClasses::printWordClasses(const char* wordClassesFile, const std::vector<WordClassIndex>& wordClasses,
                                   int verbose) const
{
  std::ofstream outF;
  outF.open(wordClassesFile, std::ios::out | std::ios::binary);
  if (!outF)
  {
    if (verbose)
      std::cerr << "Error while printing word classes file." << std::endl;
    return THOT_ERROR;
  }
  else
  {
    for (WordIndex w = 0; w < wordClasses.size(); ++w)
    {
      outF.write((char*)&w, sizeof(WordIndex));
      outF.write((char*)&wordClasses[w], sizeof(WordClassIndex));
    }
    return THOT_OK;
  }
}

bool WordClasses::printWordClassNames(const char* wordClassNamesFile,
                                      const std::unordered_map<std::string, WordClassIndex>& wordClassNames,
                                      int verbose) const
{
  std::ofstream outF(wordClassNamesFile);
  if (!outF)
  {
    if (verbose)
      std::cerr << "Error while printing word class names file." << std::endl;
    return THOT_ERROR;
  }
  else
  {
    for (auto& pair : wordClassNames)
      outF << pair.first << "\t" << pair.second << std::endl;
    return THOT_OK;
  }
}
