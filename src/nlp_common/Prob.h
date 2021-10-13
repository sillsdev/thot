#pragma once

#include <iomanip>
#include <iostream>
#include <math.h>

#define UNINIT_PROB 99
#define UNINIT_LGPROB 99

class LgProb;

class Prob
{
private:
  double x;

public:
  Prob()
  {
    x = UNINIT_PROB;
  }
  Prob(double y) : x(y)
  {
  }
  Prob(float y) : x(y)
  {
  }
  Prob(int y) : x(y)
  {
  }
  operator double() const
  {
    return x;
  }
  operator float() const
  {
    return (float)x;
  }
  Prob operator*=(double y)
  {
    x *= y;
    return *this;
  }
  Prob operator*=(Prob y)
  {
    x *= y.x;
    return *this;
  }
  Prob operator/=(double y)
  {
    x /= y;
    return *this;
  }
  Prob operator/=(Prob y)
  {
    x /= y.x;
    return *this;
  }
  Prob operator+=(double y)
  {
    x += y;
    return *this;
  }
  Prob operator+=(Prob y)
  {
    x += y.x;
    return *this;
  }
  Prob operator+(double y)
  {
    return x + y;
  }
  Prob operator+(Prob y)
  {
    return x + y.x;
  }
  Prob operator-=(double y)
  {
    x -= y;
    return *this;
  }
  Prob operator-=(Prob y)
  {
    x -= y.x;
    return *this;
  }
  Prob operator-(double y) const
  {
    return x - y;
  }
  Prob operator-(Prob y) const
  {
    return x - y.x;
  }
  Prob operator*(double y) const
  {
    return x * y;
  }
  Prob operator*(Prob y) const
  {
    return x * y.x;
  }
  Prob operator/(double y) const
  {
    return x / y;
  }
  Prob operator/(Prob y) const
  {
    return x / y.x;
  }
  bool operator==(Prob y) const
  {
    if (this->x == y.x)
      return true;
    else
      return false;
  }
  bool operator!=(Prob y) const
  {
    if (this->x != y.x)
      return true;
    else
      return false;
  }
  bool operator<(Prob y) const
  {
    if (this->x < y.x)
      return true;
    else
      return false;
  }
  bool operator>(Prob y) const
  {
    if (this->x > y.x)
      return true;
    else
      return false;
  }
  bool operator<=(Prob y) const
  {
    if (this->x <= y.x)
      return true;
    else
      return false;
  }
  bool operator>=(Prob y) const
  {
    if (this->x >= y.x)
      return true;
    else
      return false;
  }

  Prob get_p(void) const
  {
    return *this;
  }
  LgProb get_lp(void) const;
  friend std::ostream& operator<<(std::ostream& outS, const Prob& p)
  {
    outS << (double)p.x;
    return outS;
  }
  friend std::istream& operator>>(std::istream& is, Prob& p)
  {
    is >> p.x;
    return is;
  }
};

class greaterProb
{
public:
  bool operator()(const Prob& a, const Prob& b) const
  {
    if ((double)a > (double)b)
      return true;
    else
      return false;
  }
};


class LgProb
{
private:
  double x;

public:
  LgProb()
  {
    x = UNINIT_LGPROB;
  }
  LgProb(double y) : x(y)
  {
  }
  LgProb(float y) : x(y)
  {
  }
  LgProb(int y) : x(y)
  {
  }
  operator double() const
  {
    return x;
  }
  operator float() const
  {
    return (float)x;
  }
  LgProb operator*=(double y)
  {
    x *= y;
    return *this;
  }
  LgProb operator*=(LgProb y)
  {
    x *= y.x;
    return *this;
  }
  LgProb operator/=(double y)
  {
    x /= y;
    return *this;
  }
  LgProb operator/=(LgProb y)
  {
    x /= y.x;
    return *this;
  }
  LgProb operator+=(double y)
  {
    x += y;
    return *this;
  }
  LgProb operator+=(LgProb y)
  {
    x += y.x;
    return *this;
  }
  LgProb operator+(double y) const
  {
    return x + y;
  }
  LgProb operator+(LgProb y) const
  {
    return x + y.x;
  }
  LgProb operator-=(double y)
  {
    x -= y;
    return *this;
  }
  LgProb operator-=(LgProb y)
  {
    x -= y.x;
    return *this;
  }
  LgProb operator-(double y) const
  {
    return x - y;
  }
  LgProb operator-(LgProb y) const
  {
    return x - y.x;
  }
  LgProb operator*(double y) const
  {
    return x * y;
  }
  LgProb operator*(LgProb y) const
  {
    return x * y.x;
  }
  bool operator==(LgProb y) const
  {
    if (this->x == y.x)
      return true;
    else
      return false;
  }
  bool operator!=(LgProb y) const
  {
    if (this->x != y.x)
      return true;
    else
      return false;
  }
  bool operator<(LgProb y) const
  {
    if (this->x < y.x)
      return true;
    else
      return false;
  }
  bool operator>(LgProb y) const
  {
    if (this->x > y.x)
      return true;
    else
      return false;
  }
  bool operator<=(LgProb y) const
  {
    if (this->x <= y.x)
      return true;
    else
      return false;
  }
  bool operator>=(LgProb y) const
  {
    if (this->x >= y.x)
      return true;
    else
      return false;
  }

  Prob get_p(void) const
  {
    Prob p(exp(x));
    return p;
  }
  LgProb get_lp(void) const
  {
    return *this;
  }
  friend std::ostream& operator<<(std::ostream& outS, const LgProb& lp)
  {
    outS << (double)lp.x;
    return outS;
  }
};

class greaterLgProb
{
public:
  bool operator()(const LgProb& a, const LgProb& b) const
  {
    if ((double)a > (double)b)
      return true;
    else
      return false;
  }
};

