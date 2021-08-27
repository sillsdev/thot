
#pragma once

#include "nlp_common/PositionIndex.h"

#include <fstream>
#include <functional>
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
  bool getValue(unsigned int i, unsigned int j) const;
  void init(unsigned int I_dims, unsigned int J_dims);
  void putAligVec(const std::vector<PositionIndex>& aligVec);
  // Put alignment vector into word matrix.
  // aligVec[j]=0 denotes that the j'th word is not aligned.
  // j is in the range [0,J-1], i is in the range [1,I]
  bool getAligVec(std::vector<PositionIndex>& aligVec) const;
  void reset();
  void set();
  void clear();
  void set(unsigned int i, unsigned int j);
  // Set position i,j to 1. The first word has index 0
  void setValue(unsigned int i, unsigned int j, bool val);
  void transpose();
  WordAlignmentMatrix& operator=(const WordAlignmentMatrix& waMatrix);
  bool operator==(const WordAlignmentMatrix& waMatrix) const;
  WordAlignmentMatrix& flip(); // flips every bit of the matrix

  // Operations between word alignment matrices
  // Bitwise AND of two WordAligMatrix
  WordAlignmentMatrix& operator&=(const WordAlignmentMatrix& waMatrix);
  // Bitwise incl OR of two WordAligMatrix
  WordAlignmentMatrix& operator|=(const WordAlignmentMatrix& waMatrix);
  // Bitwise excl OR of two WordAligMatrix
  WordAlignmentMatrix& operator^=(const WordAlignmentMatrix& waMatrix);
  // Sum of two WordAligMatrix
  WordAlignmentMatrix& operator+=(const WordAlignmentMatrix& waMatrix);
  // Subtract waMatrix from *this
  WordAlignmentMatrix& operator-=(const WordAlignmentMatrix& waMatrix);
  // Combine two WordAligMatrix in the Och way (1999)
  WordAlignmentMatrix& symmetr1(const WordAlignmentMatrix& waMatrix);
  // Combine two WordAligMatrix in the Och way (2002, Master thesis)
  WordAlignmentMatrix& symmetr2(const WordAlignmentMatrix& waMatrix);
  // Combine two WordAligMatrix using grow
  WordAlignmentMatrix& grow(const WordAlignmentMatrix& waMatrix);
  // Combine two WordAligMatrix using grow-diag
  WordAlignmentMatrix& growDiag(const WordAlignmentMatrix& waMatrix);
  // Combine two WordAligMatrix using grow-diag-final
  WordAlignmentMatrix& growDiagFinal(const WordAlignmentMatrix& waMatrix);
  // Combine two WordAligMatrix using grow-diag-final-and
  WordAlignmentMatrix& growDiagFinalAnd(const WordAlignmentMatrix& waMatrix);

  // Predicates
  bool isRowAligned(unsigned int i) const;
  bool isColumnAligned(unsigned int j) const;
  bool isDiagonalNeighborAligned(unsigned int i, unsigned int j) const;
  bool isHorizontalNeighborAligned(unsigned int i, unsigned int j) const;
  bool isVerticalNeighborAligned(unsigned int i, unsigned int j) const;

  // Printing functions
  friend std::ostream& operator<<(std::ostream& outS, const WordAlignmentMatrix& waMatrix);
  void print(FILE* f) const;
  void wordAligAsVectors(std::vector<std::pair<unsigned int, unsigned int>>& sourceSegm,
                         std::vector<unsigned int>& targetCuts) const;

  bool** ptr();

  // Destructor
  ~WordAlignmentMatrix();

private:
  void ochGrow(std::function<bool(unsigned int, unsigned int)> growCondition, const WordAlignmentMatrix& orig,
               const WordAlignmentMatrix& other);
  void koehnGrow(std::function<bool(unsigned int, unsigned int)> growCondition, const WordAlignmentMatrix& orig,
                 const WordAlignmentMatrix& other);
  void final(std::function<bool(unsigned int, unsigned int)> pred, const WordAlignmentMatrix& adds);

  // Data members
  unsigned int I;
  unsigned int J;
  bool** matrix = nullptr;
};

std::ostream& operator<<(std::ostream& outS, const WordAlignmentMatrix& waMatrix);
