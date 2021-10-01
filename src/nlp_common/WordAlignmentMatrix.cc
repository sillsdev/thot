#include "nlp_common/WordAlignmentMatrix.h"

#include <set>

WordAlignmentMatrix::WordAlignmentMatrix()
{
  I = 0;
  J = 0;
}

WordAlignmentMatrix::WordAlignmentMatrix(unsigned int I_dims, unsigned int J_dims)
{
  I = 0;
  J = 0;
  init(I_dims, J_dims);
}

WordAlignmentMatrix::WordAlignmentMatrix(const WordAlignmentMatrix& waMatrix)
{
  unsigned int i, j;

  I = 0;
  J = 0;
  init(waMatrix.I, waMatrix.J);

  for (i = 0; i < I; ++i)
    for (j = 0; j < J; ++j)
      matrix[i][j] = waMatrix.matrix[i][j];
}

unsigned int WordAlignmentMatrix::get_I() const
{
  return I;
}

unsigned int WordAlignmentMatrix::get_J() const
{
  return J;
}

bool WordAlignmentMatrix::getValue(unsigned int i, unsigned int j) const
{
  return matrix[i][j];
}

void WordAlignmentMatrix::init(unsigned int I_dims, unsigned int J_dims)
{
  if (I != I_dims || J != J_dims)
  {
    clear();
    I = I_dims;
    J = J_dims;

    matrix = new bool*[I];
    bool* pool = new bool[(size_t)I * J]{false};
    for (unsigned int i = 0; i < I; ++i, pool += J)
      matrix[i] = pool;
  }
  else
    reset();
}

void WordAlignmentMatrix::putAligVec(const std::vector<PositionIndex>& aligVec)
{
  unsigned int j;

  if (aligVec.size() == J)
  {
    for (j = 0; j < aligVec.size(); ++j)
    {
      if (aligVec[j] > 0)
        matrix[aligVec[j] - 1][j] = true;
    }
  }
}

bool WordAlignmentMatrix::getAligVec(std::vector<PositionIndex>& aligVec) const
{
  aligVec.clear();
  for (unsigned int j = 0; j < J; ++j)
  {
    aligVec.push_back(0);
    for (unsigned int i = 0; i < I; ++i)
    {
      if (matrix[i][j])
      {
        if (aligVec[j] == 0)
          aligVec[j] = i + 1;
        else
        {
          aligVec.clear();
          return false;
        }
      }
    }
  }
  return true;
}

void WordAlignmentMatrix::reset()
{
  unsigned int i, j;

  for (i = 0; i < I; ++i)
    for (j = 0; j < J; ++j)
      matrix[i][j] = false;
}

void WordAlignmentMatrix::set()
{
  unsigned int i, j;

  for (i = 0; i < I; ++i)
    for (j = 0; j < J; ++j)
      matrix[i][j] = true;
}

void WordAlignmentMatrix::set(unsigned int i, unsigned int j)
{
  if (i < I && j < J)
    matrix[i][j] = true;
}

void WordAlignmentMatrix::setValue(unsigned int i, unsigned int j, bool val)
{
  if (i < I && j < J)
    matrix[i][j] = val;
}

void WordAlignmentMatrix::transpose()
{
  WordAlignmentMatrix wam;
  unsigned int i, j;

  wam.init(J, I);

  for (i = 0; i < I; ++i)
    for (j = 0; j < J; ++j)
    {
      wam.matrix[j][i] = matrix[i][j];
    }
  *this = wam;
}

WordAlignmentMatrix& WordAlignmentMatrix::operator=(const WordAlignmentMatrix& waMatrix)
{
  unsigned int i, j;

  init(waMatrix.I, waMatrix.J);
  for (i = 0; i < I; ++i)
    for (j = 0; j < J; ++j)
      matrix[i][j] = waMatrix.matrix[i][j];

  return *this;
}

bool WordAlignmentMatrix::operator==(const WordAlignmentMatrix& waMatrix) const
{
  unsigned int i, j;

  if (waMatrix.I != I || waMatrix.J != J)
    return false;
  for (i = 0; i < I; ++i)
    for (j = 0; j < J; ++j)
    {
      if (waMatrix.matrix[i][j] != matrix[i][j])
        return false;
    }

  return true;
}

WordAlignmentMatrix& WordAlignmentMatrix::flip()
{
  unsigned int i, j;

  for (i = 0; i < I; ++i)
    for (j = 0; j < J; ++j)
    {
      if (!matrix[i][j])
        matrix[i][j] = true;
      else
        matrix[i][j] = false;
    }
  return *this;
}

WordAlignmentMatrix& WordAlignmentMatrix::operator&=(const WordAlignmentMatrix& waMatrix)
{
  unsigned int i, j;

  if (I == waMatrix.I && J == waMatrix.J)
  {
    for (i = 0; i < I; ++i)
      for (j = 0; j < J; ++j)
      {
        if (!(matrix[i][j] && waMatrix.matrix[i][j]))
          matrix[i][j] = false;
      }
  }
  return *this;
}

WordAlignmentMatrix& WordAlignmentMatrix::operator|=(const WordAlignmentMatrix& waMatrix)
{
  unsigned int i, j;

  if (I == waMatrix.I && J == waMatrix.J)
  {
    for (i = 0; i < I; ++i)
      for (j = 0; j < J; ++j)
        if (matrix[i][j] || waMatrix.matrix[i][j])
          matrix[i][j] = true;
  }
  return *this;
}

WordAlignmentMatrix& WordAlignmentMatrix::operator^=(const WordAlignmentMatrix& waMatrix)
{
  unsigned int i, j;

  if (I == waMatrix.I && J == waMatrix.J)
  {
    for (i = 0; i < I; ++i)
      for (j = 0; j < J; ++j)
      {
        if ((matrix[i][j] && !waMatrix.matrix[i][j]) || (!matrix[i][j] && waMatrix.matrix[i][j]))
          matrix[i][j] = true;
        else
          matrix[i][j] = false;
      }
  }
  return *this;
}

WordAlignmentMatrix& WordAlignmentMatrix::operator+=(const WordAlignmentMatrix& waMatrix)
{
  unsigned int i, j;

  if (I == waMatrix.I && J == waMatrix.J)
  {
    for (i = 0; i < I; ++i)
      for (j = 0; j < J; ++j)
        matrix[i][j] = matrix[i][j] || waMatrix.matrix[i][j];
  }
  return *this;
}

WordAlignmentMatrix& WordAlignmentMatrix::operator-=(const WordAlignmentMatrix& waMatrix)
{
  unsigned int i, j;

  if (I == waMatrix.I && J == waMatrix.J)
  {
    for (i = 0; i < I; ++i)
      for (j = 0; j < J; ++j)
        if (matrix[i][j] && !waMatrix.matrix[i][j])
          matrix[i][j] = true;
  }
  return *this;
}

WordAlignmentMatrix& WordAlignmentMatrix::symmetr1(const WordAlignmentMatrix& waMatrix)
{
  if (I != waMatrix.I || J != waMatrix.J)
    return *this;

  WordAlignmentMatrix orig = *this;
  *this &= waMatrix;

  auto isBlockNeighborAligned = [&](unsigned int i, unsigned int j) {
    return isHorizontalNeighborAligned(i, j) || isVerticalNeighborAligned(i, j);
  };
  ochGrow(isBlockNeighborAligned, orig, waMatrix);

  return *this;
}

WordAlignmentMatrix& WordAlignmentMatrix::symmetr2(const WordAlignmentMatrix& waMatrix)
{
  if (I != waMatrix.I && J != waMatrix.J)
    return *this;

  auto isPriorityBlockNeighborAligned = [&](unsigned int i, unsigned int j) {
    return isHorizontalNeighborAligned(i, j) ^ isVerticalNeighborAligned(i, j);
  };
  ochGrow(isPriorityBlockNeighborAligned, *this, waMatrix);

  return *this;
}

WordAlignmentMatrix& WordAlignmentMatrix::grow(const WordAlignmentMatrix& waMatrix)
{
  if (I != waMatrix.I || J != waMatrix.J)
    return *this;

  WordAlignmentMatrix orig = *this;
  *this &= waMatrix;

  auto isBlockNeighborAligned = [&](unsigned int i, unsigned int j) {
    return isHorizontalNeighborAligned(i, j) || isVerticalNeighborAligned(i, j);
  };
  koehnGrow(isBlockNeighborAligned, orig, waMatrix);

  return *this;
}

WordAlignmentMatrix& WordAlignmentMatrix::growDiag(const WordAlignmentMatrix& waMatrix)
{
  if (I != waMatrix.I || J != waMatrix.J)
    return *this;

  WordAlignmentMatrix orig = *this;
  *this &= waMatrix;

  auto isBlockOrDiagNeighborAligned = [&](unsigned int i, unsigned int j) {
    return isHorizontalNeighborAligned(i, j) || isVerticalNeighborAligned(i, j) || isDiagonalNeighborAligned(i, j);
  };
  koehnGrow(isBlockOrDiagNeighborAligned, orig, waMatrix);

  return *this;
}

WordAlignmentMatrix& WordAlignmentMatrix::growDiagFinal(const WordAlignmentMatrix& waMatrix)
{
  if (I != waMatrix.I || J != waMatrix.J)
    return *this;

  WordAlignmentMatrix orig = *this;
  *this &= waMatrix;

  auto isBlockOrDiagNeighborAligned = [&](unsigned int i, unsigned int j) {
    return isHorizontalNeighborAligned(i, j) || isVerticalNeighborAligned(i, j) || isDiagonalNeighborAligned(i, j);
  };
  koehnGrow(isBlockOrDiagNeighborAligned, orig, waMatrix);

  auto isOneOrBothUnaligned = [&](unsigned int i, unsigned int j) { return !isRowAligned(i) || !isColumnAligned(j); };
  final(isOneOrBothUnaligned, orig);
  final(isOneOrBothUnaligned, waMatrix);

  return *this;
}

WordAlignmentMatrix& WordAlignmentMatrix::growDiagFinalAnd(const WordAlignmentMatrix& waMatrix)
{
  if (I != waMatrix.I || J != waMatrix.J)
    return *this;

  WordAlignmentMatrix orig = *this;
  *this &= waMatrix;

  auto isBlockOrDiagNeighborAligned = [&](unsigned int i, unsigned int j) {
    return isHorizontalNeighborAligned(i, j) || isVerticalNeighborAligned(i, j) || isDiagonalNeighborAligned(i, j);
  };
  koehnGrow(isBlockOrDiagNeighborAligned, orig, waMatrix);

  auto isBothUnaligned = [&](unsigned int i, unsigned int j) { return !isRowAligned(i) && !isColumnAligned(j); };
  final(isBothUnaligned, orig);
  final(isBothUnaligned, waMatrix);

  return *this;
}

bool WordAlignmentMatrix::isDiagonalNeighborAligned(unsigned int i, unsigned int j) const
{
  if (i < I - 1 && j < J - 1)
    if (matrix[i + 1][j + 1])
      return true;
  if (i > 0 && j < J - 1)
    if (matrix[i - 1][j + 1])
      return true;
  if (i < I - 1 && j > 0)
    if (matrix[i + 1][j - 1])
      return true;
  if (i > 0 && j > 0)
    if (matrix[i - 1][j - 1])
      return true;

  return false;
}

bool WordAlignmentMatrix::isHorizontalNeighborAligned(unsigned int i, unsigned int j) const
{
  if (j > 0)
    if (matrix[i][j - 1])
      return true;
  if (j < J - 1)
    if (matrix[i][j + 1])
      return true;

  return false;
}

bool WordAlignmentMatrix::isVerticalNeighborAligned(unsigned int i, unsigned int j) const
{
  if (i > 0)
    if (matrix[i - 1][j])
      return true;
  if (i < I - 1)
    if (matrix[i + 1][j])
      return true;

  return false;
}

bool WordAlignmentMatrix::isColumnAligned(unsigned int j) const
{
  for (unsigned int i = 0; i < I; ++i)
    if (matrix[i][j])
      return true;

  return false;
}

bool WordAlignmentMatrix::isRowAligned(unsigned int i) const
{
  for (unsigned int j = 0; j < J; ++j)
    if (matrix[i][j])
      return 1;

  return 0;
}

void WordAlignmentMatrix::clear(void)
{
  if (I > 0)
  {
    delete[] matrix[0];
    delete[] matrix;
    matrix = nullptr;
  }
  I = 0;
  J = 0;
}

std::ostream& operator<<(std::ostream& outS, const WordAlignmentMatrix& waMatrix)
{
  unsigned int j;
  int i;

  for (i = (int)waMatrix.I - 1; i >= 0; --i)
  {
    for (j = 0; j < waMatrix.J; ++j)
      outS << (unsigned int)waMatrix.matrix[i][j] << " ";
    outS << std::endl;
  }
  return outS;
}

void WordAlignmentMatrix::print(FILE* f) const
{
  unsigned int j;
  int i;

  for (i = (int)this->I - 1; i >= 0; --i)
  {
    for (j = 0; j < this->J; ++j)
      fprintf(f, "%d ", this->matrix[i][j]);
    fprintf(f, "\n");
  }
}

void WordAlignmentMatrix::wordAligAsVectors(std::vector<std::pair<unsigned int, unsigned int>>& sourceSegm,
                                            std::vector<unsigned int>& targetCuts) const
{
  std::pair<unsigned int, unsigned int> prevIntPair, intPair;
  unsigned int i, j;

  targetCuts.clear();

  prevIntPair.first = 0;
  prevIntPair.second = 0;
  for (i = 0; i < I; ++i)
  {
    intPair.first = 0;
    intPair.second = 0;
    for (j = 0; j < J; ++j)
    {
      if (matrix[i][j] && intPair.first == 0)
        intPair.first = j + 1;
      if (!matrix[i][j] && intPair.first != 0 && intPair.second == 0)
        intPair.second = j;
    }
    if (intPair.second == 0)
      intPair.second = j;
    if (intPair != prevIntPair)
    {
      sourceSegm.push_back(intPair);
      if (i != 0)
        targetCuts.push_back(i);
      prevIntPair = intPair;
    }
  }
  targetCuts.push_back(i);
}

bool** WordAlignmentMatrix::ptr()
{
  return matrix;
}

WordAlignmentMatrix::~WordAlignmentMatrix()
{
  clear();
}

void WordAlignmentMatrix::ochGrow(std::function<bool(unsigned int, unsigned int)> growCondition,
                                  const WordAlignmentMatrix& orig, const WordAlignmentMatrix& other)
{
  bool added;
  do
  {
    added = false;
    for (unsigned int i = 0; i < I; ++i)
    {
      for (unsigned int j = 0; j < J; ++j)
      {
        if ((other.getValue(i, j) || orig.getValue(i, j)) && !getValue(i, j))
        {
          if (!isRowAligned(i) && !isColumnAligned(j))
          {
            set(i, j);
            added = true;
          }
          else if (growCondition(i, j))
          {
            set(i, j);
            added = true;
          }
        }
      }
    }

  } while (added);
}

void WordAlignmentMatrix::koehnGrow(std::function<bool(unsigned int, unsigned int)> growCondition,
                                    const WordAlignmentMatrix& orig, const WordAlignmentMatrix& other)
{
  std::set<std::pair<unsigned int, unsigned int>> p;
  for (unsigned int i = 0; i < I; ++i)
  {
    for (unsigned int j = 0; j < J; ++j)
    {
      if ((orig.getValue(i, j) || other.getValue(i, j)) && !getValue(i, j))
        p.insert(std::make_pair(i, j));
    }
  }

  bool keepGoing = !p.empty();
  while (keepGoing)
  {
    keepGoing = false;
    std::set<std::pair<unsigned int, unsigned int>> added;
    for (auto pair : p)
    {
      unsigned int i = pair.first;
      unsigned int j = pair.second;
      if ((!isRowAligned(i) || !isColumnAligned(j)) && growCondition(i, j))
      {
        set(i, j);
        added.insert(std::make_pair(i, j));
        keepGoing = true;
      }
    }
    for (auto pair : added)
      p.erase(pair);
  }
}

void WordAlignmentMatrix::final(std::function<bool(unsigned int, unsigned int)> pred, const WordAlignmentMatrix& adds)
{
  for (unsigned int i = 0; i < I; ++i)
  {
    for (unsigned int j = 0; j < J; ++j)
    {
      if (adds.getValue(i, j) && !getValue(i, j) && pred(i, j))
        set(i, j);
    }
  }
}
