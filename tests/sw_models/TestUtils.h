#include "sw_models/AlignmentModel.h"
#include "sw_models/IncrAlignmentModel.h"

#include <string>
#include <unordered_set>
#include <vector>

std::pair<unsigned int, unsigned int> addSentencePair(AlignmentModel& model, const std::string& srcSentence,
                                                      const std::string& trgSentence);
void addTrainingData(AlignmentModel& model);
void addTrainingDataWordClasses(AlignmentModel& model);
void addSrcWordClass(AlignmentModel& model, const std::string& c, const std::unordered_set<std::string>& words);
void addTrgWordClass(AlignmentModel& model, const std::string& c, const std::unordered_set<std::string>& words);
void train(AlignmentModel& model, int numIters = 1);
void incrTrain(IncrAlignmentModel& model, std::pair<unsigned int, unsigned int> range, int numIters = 1);
