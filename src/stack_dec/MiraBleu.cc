


//--------------- Include files --------------------------------------

#include "MiraBleu.h"
#include "bleu.h"

#include <sstream>
//--------------- KbMiraLlWu class functions

//---------------------------------------
double MiraBleu::scoreFromStats(Vector<unsigned int>& stats){
  double bp;
  if (stats[0] < stats[1])
    bp = (double)exp((double)1-(double)stats[1]/stats[0]);
  else bp = 1;

  double log_aux = 0;
  for (unsigned int sz=1; sz<=4; sz++) {
    unsigned int prec = stats[sz*2];
    unsigned int total = stats[sz*2+1];
    if (total == 0) log_aux += 1;
    else            log_aux += (double)my_log((double)prec/total);
  }
  log_aux /= 4;
  return bp * (double)exp(log_aux);
}

//---------------------------------------
void MiraBleu::statsForSentence(const Vector<std::string>& candidate_tokens,
                                const Vector<std::string>& reference_tokens,
                                Vector<unsigned int>& stats)
{
  stats.clear();

  unsigned int prec, total;
  stats.push_back(candidate_tokens.size());
  stats.push_back(reference_tokens.size());
  for (unsigned int sz=1; sz<=4; sz++) {
    prec_n(reference_tokens, candidate_tokens, sz, prec, total);
    stats.push_back(prec);
    stats.push_back(total);
  }
}

//---------------------------------------
void MiraBleu::split(const std::string& sentence,
                     Vector<std::string>& tokens)
{
  std::string item;
  std::stringstream ss(sentence);
  while (std::getline(ss, item, ' '))
    tokens.push_back(item);
}

//---------------------------------------
void MiraBleu::sentBackgroundScore(const std::string& candidate,
                                   const std::string& reference,
                                   double& bleu,
                                   Vector<unsigned int>& sentStats)
{
  Vector<std::string> candidate_tokens, reference_tokens;
  split(candidate, candidate_tokens);
  split(reference, reference_tokens);

  statsForSentence(candidate_tokens, reference_tokens, sentStats);

  Vector<unsigned int> stats;
  for (unsigned int i=0; i<N_STATS; i++)
    stats.push_back(sentStats[i] + backgroundBleu[i]);

  // scale bleu to roughly typical margins
  bleu = scoreFromStats(stats) * reference_tokens.size();
}

//---------------------------------------
void MiraBleu::Score(const Vector<std::string>& candidates,
                       const Vector<std::string>& references,
                       double& bleu)
{
  Vector<unsigned int> corpusStats(N_STATS, 0);
  for (unsigned int i=0; i<candidates.size(); i++) {
    Vector<std::string> candidate_tokens, reference_tokens;
    split(candidates[i], candidate_tokens);
    split(references[i], reference_tokens);

    Vector<unsigned int> stats;
    statsForSentence(candidate_tokens, reference_tokens, stats);

    for (unsigned int i=0; i<N_STATS; i++)
      corpusStats[i] += stats[i];
  }
  bleu = scoreFromStats(corpusStats);

}

//---------------------------------------
void MiraBleu::sentScore(const std::string& candidate,
                         const std::string& reference,
                         double& bleu)
{
  Vector<std::string> candidate_tokens, reference_tokens;
  split(candidate, candidate_tokens);
  split(reference, reference_tokens);

  Vector<unsigned int> stats;
  statsForSentence(candidate_tokens, reference_tokens, stats);

  for (unsigned int i=0; i<N_STATS; i++)
    stats[i] += 1;

  bleu = scoreFromStats(stats);
}
