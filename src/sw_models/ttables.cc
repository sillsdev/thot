#include "ttables.h"

#include <cmath>
#include <string>
#include <fstream>

void TTable::deserializeLogProbsFromText(const std::string& filename, SingleWordVocab& vocab)
{
  std::ifstream in(filename);
  int c = 0;
  std::string e, f;
  double p;
  while(in)
  {
    in >> e >> f >> p;
    if (e.empty()) break;
    ++c;
    unsigned ie = vocab.stringToSrcWordIndex(e);
    if (ie >= ttable.size()) ttable.resize((size_t)ie + 1);
    ttable[ie][vocab.stringToTrgWordIndex(f)] = std::exp(p);
  }
  frozen_ = true;
  probs_initialized_ = true;
}

