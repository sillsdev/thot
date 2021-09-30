#include "sw_models/LightSentenceHandler.h"

#include "nlp_common/ErrorDefs.h"

LightSentenceHandler::LightSentenceHandler()
{
  nsPairsInFiles = 0;
  countFileExists = false;
  currFileSentIdx = 0;
}

bool LightSentenceHandler::readSentencePairs(const char* srcFileName, const char* trgFileName,
                                             const char* sentCountsFile,
                                             std::pair<unsigned int, unsigned int>& sentRange, int verbose /*=0*/)
{
  // Clear sentence handler
  if (verbose)
    std::cerr << "Initializing sentence handler..." << std::endl;
  clear();

  // Fill first field of sentRange
  sentRange.first = 0;

  // Open source file
  if (awkSrc.open(srcFileName) == THOT_ERROR)
  {
    if (verbose)
      std::cerr << "Error in source language file: " << srcFileName << std::endl;
    return THOT_ERROR;
  }
  else
  {
    // Open target file
    if (awkTrg.open(trgFileName) == THOT_ERROR)
    {
      if (verbose)
        std::cerr << "Error in target language file: " << trgFileName << std::endl;
      return THOT_ERROR;
    }
    else
    {
      // Open file with sentence counts
      if (strlen(sentCountsFile) == 0)
      {
        // sentCountsFile is empty
        countFileExists = false;
      }
      else
      {
        // sentCountsFile is not empty
        if (awkSrcTrgC.open(sentCountsFile) == THOT_ERROR)
        {
          if (verbose)
            std::cerr << "File with sentence counts " << sentCountsFile << " does not exist" << std::endl;
          countFileExists = false;
        }
        else
          countFileExists = true;
      }

      // Read sentence pairs
      if (verbose)
      {
        std::cerr << "Reading sentence pairs from files: " << srcFileName << " and " << trgFileName << std::endl;
        if (countFileExists)
          std::cerr << "Reading sentence pair counts from file " << sentCountsFile << std::endl;
      }

      while (awkSrc.getln())
      {
        if (!awkTrg.getln())
        {
          if (verbose)
            std::cerr << "Error: the number of source and target sentences differ!" << std::endl;
          return THOT_ERROR;
        }

        // Display warnings if sentences are empty
        if (verbose)
        {
          if (awkSrc.NF == 0)
            std::cerr << "Warning: source sentence " << nsPairsInFiles << " is empty" << std::endl;
          if (awkTrg.NF == 0)
            std::cerr << "Warning: target sentence " << nsPairsInFiles << " is empty" << std::endl;
        }

        nsPairsInFiles += 1;
      }
      // Print statistics
      if (verbose && nsPairsInFiles > 0)
        std::cerr << "#Sentence pairs in files: " << nsPairsInFiles << std::endl;
    }
    // Fill second field of sentRange
    sentRange.second = nsPairsInFiles - 1;

    // Rewind files
    rewindFiles();

    return THOT_OK;
  }
}

void LightSentenceHandler::rewindFiles()
{
  // Rewind files
  awkSrc.rwd();
  awkTrg.rwd();
  awkSrcTrgC.rwd();

  // Read first entry
  getNextLineFromFiles();

  // Reset currFileSentIdx
  currFileSentIdx = 0;
}

std::pair<unsigned int, unsigned int> LightSentenceHandler::addSentencePair(std::vector<std::string> srcSentStr,
                                                                            std::vector<std::string> trgSentStr,
                                                                            Count c, int verbose)
{
  unsigned int index = nsPairsInFiles + sentPairCont.size();
  // Fill sentRange information
  std::pair<unsigned int, unsigned int> sentRange{index, index};
  // add to sentPairCont
  sentPairCont.push_back(std::make_pair(srcSentStr, trgSentStr));
  // add to sentPairCount
  sentPairCount.push_back(c);

  if (verbose)
  {
    // Display warnings if sentences are empty
    if (srcSentStr.empty())
      std::cerr << "Warning: source sentence " << sentRange.first << " is empty" << std::endl;
    if (trgSentStr.empty())
      std::cerr << "Warning: target sentence " << sentRange.first << " is empty" << std::endl;
  }
  return sentRange;
}

unsigned int LightSentenceHandler::numSentencePairs()
{
  return nsPairsInFiles + sentPairCont.size();
}

int LightSentenceHandler::getSentencePair(unsigned int n, std::vector<std::string>& srcSentStr,
                                          std::vector<std::string>& trgSentStr, Count& c)
{
  if (n >= numSentencePairs())
    return THOT_ERROR;
  else
  {
    if (n < nsPairsInFiles)
    {
      return nthSentPairFromFiles(n, srcSentStr, trgSentStr, c);
    }
    else
    {
      size_t vecIdx = n - nsPairsInFiles;

      srcSentStr = sentPairCont[vecIdx].first;

      trgSentStr = sentPairCont[vecIdx].second;

      c = sentPairCount[vecIdx];

      return THOT_OK;
    }
  }
}

int LightSentenceHandler::nthSentPairFromFiles(unsigned int n, std::vector<std::string>& srcSentStr,
                                               std::vector<std::string>& trgSentStr, Count& c)

{
  // Check if entry is contained in files
  if (n >= nsPairsInFiles)
    return THOT_ERROR;

  // Find corresponding entries
  if (currFileSentIdx > n)
    rewindFiles();

  if (currFileSentIdx != n)
  {
    while (getNextLineFromFiles())
    {
      if (currFileSentIdx == n)
        break;
    }
  }

  // Reset variables
  srcSentStr.clear();
  trgSentStr.clear();

  // Extract information
  for (unsigned int i = 1; i <= awkSrc.NF; ++i)
  {
    srcSentStr.push_back(awkSrc.dollar(i));
  }
  for (unsigned int i = 1; i <= awkTrg.NF; ++i)
  {
    trgSentStr.push_back(awkTrg.dollar(i));
  }

  if (countFileExists)
  {
    c = atof(awkSrcTrgC.dollar(1).c_str());
  }
  else
  {
    c = 1;
  }

  return THOT_OK;
}

bool LightSentenceHandler::getNextLineFromFiles()
{
  bool ret;

  ret = awkSrc.getln();
  if (ret == false)
    return false;

  ret = awkTrg.getln();
  if (ret == false)
    return false;

  if (countFileExists)
  {
    ret = awkSrcTrgC.getln();
    if (ret == false)
      return false;
  }

  ++currFileSentIdx;

  return true;
}

int LightSentenceHandler::getSrcSentence(unsigned int n, std::vector<std::string>& srcSentStr)
{
  std::vector<std::string> trgSentStr;
  Count c;

  int ret = getSentencePair(n, srcSentStr, trgSentStr, c);

  return ret;
}

int LightSentenceHandler::getTrgSentence(unsigned int n, std::vector<std::string>& trgSentStr)
{
  std::vector<std::string> srcSentStr;
  Count c;

  int ret = getSentencePair(n, srcSentStr, trgSentStr, c);

  return ret;
}

int LightSentenceHandler::getCount(unsigned int n, Count& c)
{
  std::vector<std::string> srcSentStr;
  std::vector<std::string> trgSentStr;

  int ret = getSentencePair(n, srcSentStr, trgSentStr, c);

  return ret;
}

bool LightSentenceHandler::printSentencePairs(const char* srcSentFile, const char* trgSentFile,
                                              const char* sentCountsFile)
{
  std::ofstream srcOutF;
  std::ofstream trgOutF;
  std::ofstream countsOutF;

  // Open file with source sentences
  srcOutF.open(srcSentFile, std::ios::binary);
  if (!srcOutF)
  {
    std::cerr << "Error while printing file with source sentences." << std::endl;
    return THOT_ERROR;
  }

  // Open file with target sentences
  trgOutF.open(trgSentFile, std::ios::binary);
  if (!trgOutF)
  {
    std::cerr << "Error while printing file with target sentences." << std::endl;
    return THOT_ERROR;
  }

  // Open file with sentence counts
  countsOutF.open(sentCountsFile, std::ios::binary);
  if (!countsOutF)
  {
    std::cerr << "Error while printing file with sentence counts." << std::endl;
    return THOT_ERROR;
  }

  for (unsigned int n = 0; n < numSentencePairs(); ++n)
  {
    std::vector<std::string> srcSentStr;
    std::vector<std::string> trgSentStr;
    Count c;

    getSentencePair(n, srcSentStr, trgSentStr, c);

    // print source sentence
    for (unsigned int j = 0; j < srcSentStr.size(); ++j)
    {
      srcOutF << srcSentStr[j];
      if (j < srcSentStr.size() - 1)
        srcOutF << " ";
    }
    srcOutF << std::endl;

    // print target sentence
    for (unsigned int j = 0; j < trgSentStr.size(); ++j)
    {
      trgOutF << trgSentStr[j];
      if (j < trgSentStr.size() - 1)
        trgOutF << " ";
    }
    trgOutF << std::endl;

    // print count
    countsOutF << c << std::endl;
  }

  // Close output streams
  srcOutF.close();
  trgOutF.close();
  countsOutF.close();

  return THOT_OK;
}

void LightSentenceHandler::clear()
{
  sentPairCont.clear();
  sentPairCount.clear();
  nsPairsInFiles = 0;
  awkSrc.close();
  awkTrg.close();
  awkSrcTrgC.close();
  countFileExists = false;
  currFileSentIdx = 0;
}
