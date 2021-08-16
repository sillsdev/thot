#include "sw_models/AlignmentModel.h"
#include "sw_models/IncrAlignmentModel.h"

#include <string>
#include <vector>

std::pair<unsigned int, unsigned int> addSentencePair(AlignmentModel& model, const std::string& srcSentence,
                                                      const std::string& trgSentence);
void addTrainingData(AlignmentModel& model);
void train(AlignmentModel& model, int numIters = 1);
void incrTrain(IncrAlignmentModel& model, std::pair<unsigned int, unsigned int> range, int numIters = 1);
LgProb obtainBestAlignment(AlignmentModel& model, const std::string& srcSentence, const std::string& trgSentence,
                           std::vector<PositionIndex>& alignment);
