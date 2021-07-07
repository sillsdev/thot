#pragma once

#include <cassert>
#include <ostream>
#include <vector>

template <class T, class Y = std::vector<T>>
class Matrix
{
public:
  Y p;
  //  short h1, h2;
  unsigned int h1, h2;

public:
  Matrix(unsigned int _h1, unsigned int _h2) : p(_h1 * _h2), h1(_h1), h2(_h2)
  {
  }
  Matrix(unsigned int _h1, unsigned int _h2, const T& _init) : p(_h1 * _h2, _init), h1(_h1), h2(_h2)
  {
  }
  Matrix() : h1(0), h2(0)
  {
  }
  inline T& operator()(unsigned int i, unsigned int j)
  {
    assert(i < h1);
    assert(j < h2);
    return p[i * h2 + j];
  }
  inline const T& operator()(unsigned int i, unsigned int j) const
  {
    assert(i < h1);
    assert(j < h2);
    return p[i * h2 + j];
  }
  inline T get(unsigned int i, unsigned int j)
  {
    assert(i < h1);
    assert(j < h2);
    return p[i * h2 + j];
  }
  inline void set(unsigned int i, unsigned int j, T x)
  {
    assert(i < h1);
    assert(j < h2);
    p[i * h2 + j] = x;
  }
  inline const T get(unsigned int i, unsigned int j) const
  {
    assert(i < h1);
    assert(j < h2);
    return p[i * h2 + j];
  }
  inline unsigned int getLen1() const
  {
    return h1;
  }
  inline unsigned int getLen2() const
  {
    return h2;
  }

  inline T* begin()
  {
    if (h1 == 0 || h2 == 0)
      return 0;
    return &(p[0]);
  }
  inline T* end()
  {
    if (h1 == 0 || h2 == 0)
      return 0;
    return &(p[0]) + p.size();
  }

  inline const T* begin() const
  {
    return p.begin();
  }
  inline const T* end() const
  {
    return p.end();
  }

  friend std::ostream& operator<<(std::ostream& out, const Matrix<T, Y>& ar)
  {
    for (unsigned int i = 0; i < ar.getLen1(); i++)
    {
      // out << i << ": ";
      for (unsigned int j = 0; j < ar.getLen2(); j++)
        out << ar(i, j) << ' ';
      out << '\n';
    }
    return out << std::endl;
  }
  inline void resize(unsigned int a, unsigned int b)
  {
    if (!(a == h1 && b == h2))
    {
      h1 = a;
      h2 = b;
      p.resize(h1 * h2);
    }
  }
  inline void resize(unsigned int a, unsigned int b, const T& t)
  {
    if (!(a == h1 && b == h2))
    {
      h1 = a;
      h2 = b;
      p.resize(h1 * h2);
      fill(p.begin(), p.end(), t);
    }
  }
};
