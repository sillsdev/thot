

#include "nlp_common/WordAlignmentMatrix.h"

#include <string>
#include <vector>

void printAlignmentInGIZAFormat(std::ostream& outS, const std::vector<std::string>& ns,
                                const std::vector<std::string>& t, WordAlignmentMatrix waMatrix, const char* header);
void printAlignmentInMyFormat(std::ostream& outS, const std::vector<std::string>& ns, const std::vector<std::string>& t,
                              WordAlignmentMatrix waMatrix, unsigned int numReps = 1);

void printAlignmentInGIZAFormat(FILE* outf, const std::vector<std::string>& ns, const std::vector<std::string>& t,
                                WordAlignmentMatrix waMatrix, const char* header);
void printAlignmentInMyFormat(FILE* outf, const std::vector<std::string>& ns, const std::vector<std::string>& t,
                              WordAlignmentMatrix waMatrix, unsigned int numReps = 1);
