#pragma once

#include <assert.h>
#include <math.h>

struct Md
{
  static double digamma(double x)
  {
    double result = 0, xx, xx2, xx4;
    for (; x < 7; ++x)
      result -= 1 / x;
    x -= 1.0 / 2.0;
    xx = 1.0 / x;
    xx2 = xx * xx;
    xx4 = xx2 * xx2;
    result +=
        log(x) + (1. / 24.) * xx2 - (7.0 / 960.0) * xx4 + (31.0 / 8064.0) * xx4 * xx2 - (127.0 / 30720.0) * xx4 * xx4;
    return result;
  }
  static inline double log_poisson(unsigned x, const double& lambda)
  {
    assert(lambda > 0.0);
    return log(lambda) * x - lgamma(x + 1) - lambda;
  }
};

