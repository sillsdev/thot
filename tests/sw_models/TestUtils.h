#include "sw_models/BaseSwAligModel.h"
#include "sw_models/_incrSwAligModel.h"

#include <string>
#include <vector>

std::pair<unsigned int, unsigned int> addSentencePair(BaseSwAligModel& model, const std::string& srcSentence,
                                                      const std::string& trgSentence);
void addTrainingData(BaseSwAligModel& model);
void train(BaseSwAligModel& model, int numIters = 1);
void incrTrain(_incrSwAligModel& model, std::pair<unsigned int, unsigned int> range, int numIters = 1);
LgProb obtainBestAlignment(BaseSwAligModel& model, const std::string& srcSentence, const std::string& trgSentence,
                           std::vector<PositionIndex>& alignment);
