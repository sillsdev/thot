
#pragma once

#include "nlp_common/PositionIndex.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

class WordAlignmentMatrix
{
public:
  // Constructors
  WordAlignmentMatrix();
  WordAlignmentMatrix(unsigned int I_dims, unsigned int J_dims);
  WordAlignmentMatrix(const WordAlignmentMatrix& waMatrix);

  // Basic operations
  unsigned int get_I() const;
  unsigned int get_J() const;
  unsigned int getValue(unsigned int i, unsigned int j) const;
  void init(unsigned int I_dims, unsigned int J_dims);
  void putAligVec(std::vector<PositionIndex> aligVec);
  // Put alignment vector into word matrix.
  // aligVec[j]=0 denotes that the j'th word is not aligned.
  // j is in the range [0,J-1], i is in the range [1,I]
  bool getAligVec(std::vector<PositionIndex>& aligVec) const;
  void reset(void);
  void set(void);
  void clear(void);
  void set(unsigned int i, unsigned int j);
  // Set position i,j to 1. The first word has index 0
  void setValue(unsigned int i, unsigned int j, unsigned int val);
  void transpose(void);
  WordAlignmentMatrix& operator=(const WordAlignmentMatrix& waMatrix);
  bool operator==(const WordAlignmentMatrix& waMatrix);
  WordAlignmentMatrix& flip(void); // flips every bit of the matrix
  std::vector<std::pair<unsigned int, unsigned int>> obtainAdjacentCells(unsigned int i, unsigned int j);

  // Operations between word alignment matrices
  WordAlignmentMatrix& operator&=(const WordAlignmentMatrix& waMatrix);
  // Bitwise AND of two WordAligMatrix
  WordAlignmentMatrix& operator|=(const WordAlignmentMatrix& waMatrix);
  // Bitwise incl OR of two WordAligMatrix
  WordAlignmentMatrix& operator^=(const WordAlignmentMatrix& waMatrix);
  // Bitwise excl OR of two WordAligMatrix
  WordAlignmentMatrix& operator+=(const WordAlignmentMatrix& waMatrix);
  // Sum of two WordAligMatrix
  WordAlignmentMatrix& operator-=(const WordAlignmentMatrix& waMatrix);
  // Sustract waMatrix from *this
  WordAlignmentMatrix& symmetr1(const WordAlignmentMatrix& waMatrix);
  // Combine two WordAligMatrix in the Och way (1999)
  WordAlignmentMatrix& symmetr2(const WordAlignmentMatrix& waMatrix);
  // Combine two WordAligMatrix in the Och way (2002, Master thesis)
  WordAlignmentMatrix& growDiagFinal(const WordAlignmentMatrix& waMatrix);
  // Combine two WordAligMatrix using grow-diag-final

  // Predicates
  bool jAligned(unsigned int j) const;
  bool iAligned(unsigned int i) const;
  bool ijInNeighbourhood(unsigned int i, unsigned int j);
  bool ijHasHorizNeighbours(unsigned int i, unsigned int j);
  bool ijHasVertNeighbours(unsigned int i, unsigned int j);

  // Printing functions
  friend std::ostream& operator<<(std::ostream& outS, const WordAlignmentMatrix& waMatrix);
  void print(FILE* f);
  void wordAligAsVectors(std::vector<std::pair<unsigned int, unsigned int>>& sourceSegm,
                         std::vector<unsigned int>& targetCuts);

  // Destructor
  ~WordAlignmentMatrix();

private:
  // Data members
  unsigned int I;
  unsigned int J;
  unsigned int** matrix = NULL;
};

std::ostream& operator<<(std::ostream& outS, const WordAlignmentMatrix& waMatrix);
