#pragma once

#include "nlp_common/AwkInputStream.h"
#include "nlp_common/WordAlignmentMatrix.h"
#include "phrase_models/PhraseDefs.h"

#include <iostream>
#include <string.h>
#include <string>
#include <vector>

#define ENDOFFILE 0
#define NO_ERRORS 1
#define ALIG_OP_FILE_ERROR -1
#define GIZA_ALIG_FILE_FORMAT 0
#define ALIG_OP_FILE_FORMAT 1

struct aligOpDescription
{
  std::string aligOperator;
  std::string GizaAligFile;
};

/*
 * AlignmentExtractor class: class for extracting
 * sentence pair alignments from a GIZA xxx.A3.final file
 */
class AlignmentExtractor
{
public:
  // Constructor
  AlignmentExtractor(void);
  AlignmentExtractor(const AlignmentExtractor& alExt);
  AlignmentExtractor& operator=(const AlignmentExtractor& alExt);

  // Functions to manipulate files
  bool open(const char* str, unsigned int _fileFormat = GIZA_ALIG_FILE_FORMAT);
  bool open_stream(FILE* stream, unsigned int _fileFormat = GIZA_ALIG_FILE_FORMAT);
  void close(void);
  bool rewind(void);
  bool getNextAlignment(void);

  // Functions to access alignment information
  std::vector<std::string> get_ns(void);
  std::vector<std::string> get_s(void);
  std::vector<std::string> get_t(void);
  WordAlignmentMatrix get_wamatrix(void);
  float get_numReps(void);

  // Functions to operate alignments
  void transposeAlig(void);
  bool join(const char* GizaAligFileName, const char* outFileName, bool transpose = 0, bool verbose = 0);
  // joins the alignment matrices given in the GIZA file (one to
  // one correspondence between alignments is assumed and prints
  // it to outF.
  bool intersect(const char* GizaAligFileName, const char* outFileName, bool transpose = 0, bool verbose = 0);
  // intersects the alignment matrixes
  bool sum(const char* GizaAligFileName, const char* outFileName, bool transpose = 0, bool verbose = 0);
  // Obtains the sum of the alignment matrixes
  bool symmetr1(const char* GizaAligFileName, const char* outFileName, bool transpose = 0, bool verbose = 0);
  bool symmetr2(const char* GizaAligFileName, const char* outFileName, bool transpose = 0, bool verbose = 0);
  bool growDiagFinal(const char* GizaAligFileName, const char* outFileName, bool transpose = 0, bool verbose = 0);

  // Destructor
  ~AlignmentExtractor();

private:
  std::vector<std::string> ns;
  std::vector<std::string> t;
  WordAlignmentMatrix wordAligMatrix;
  float numReps;
  unsigned int fileFormat;
  FILE* fileStream;
  AwkInputStream awkInpStrm;

  bool getNextAlignInGIZAFormat(void);
  bool getNextAlignInAlignOpFormat(void);
};

std::ostream& operator<<(std::ostream& outS, AlignmentExtractor& ae);
