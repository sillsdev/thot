#include "sw_models/Aligner.h"

#include <memory>

enum class SymmetrizationHeuristic
{
  None,
  Union,
  Intersection,
  Och,
  Grow,
  GrowDiag,
  GrowDiagFinal,
  GrowDiagFinalAnd
};

class SymmetrizedAligner : public virtual Aligner
{
public:
  SymmetrizedAligner(std::shared_ptr<Aligner> directAligner, std::shared_ptr<Aligner> inverseAligner);

  void setHeuristic(SymmetrizationHeuristic value);
  SymmetrizationHeuristic getHeuristic() const;

  LgProb getBestAlignment(const char* srcSentence, const char* trgSentence, WordAlignmentMatrix& bestWaMatrix) override;
  LgProb getBestAlignment(const std::vector<std::string>& srcSentence, const std::vector<std::string>& trgSentence,
                          WordAlignmentMatrix& bestWaMatrix) override;
  LgProb getBestAlignment(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                          WordAlignmentMatrix& bestWaMatrix) override;

  WordIndex stringToSrcWordIndex(std::string s) const override;
  std::vector<WordIndex> strVectorToSrcIndexVector(std::vector<std::string> s) override;

  WordIndex stringToTrgWordIndex(std::string t) const override;
  std::vector<WordIndex> strVectorToTrgIndexVector(std::vector<std::string> t) override;

  virtual ~SymmetrizedAligner()
  {
  }

private:
  std::shared_ptr<Aligner> directAligner;
  std::shared_ptr<Aligner> inverseAligner;

  SymmetrizationHeuristic heuristic = SymmetrizationHeuristic::Och;
};
