/*
thot package for statistical machine translation
Copyright (C) 2013 Daniel Ortiz-Mart\'inez

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program; If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * @file OrderedVector.h
 *
 * @brief Implements an ordered vector that allows to search elements
 * with logarithmic cost.
 */

#pragma once

#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <vector>

template <class KEY, class DATA, class KEY_ORDER_REL = std::less<KEY>>
class OrderedVector
{
public:
  using iterator = typename std::vector<std::pair<KEY, DATA>>::iterator;
  using const_iterator = typename std::vector<std::pair<KEY, DATA>>::const_iterator;

  OrderedVector() : data(), compare()
  {
  }

  OrderedVector(const OrderedVector<KEY, DATA, KEY_ORDER_REL>& ov) : data(ov.data), compare()
  {
  }

  OrderedVector<KEY, DATA, KEY_ORDER_REL>& operator=(const OrderedVector<KEY, DATA, KEY_ORDER_REL>& ov)
  {
    if (&ov != this)
      data = ov.data;
    return *this;
  }

  DATA* push(const KEY& k, const DATA& d)
  {
    iterator iter = std::lower_bound(data.begin(), data.end(), k, compare);
    if (iter == data.end() || compare(k, *iter))
    {
      iter = data.insert(iter, std::make_pair(k, d));
      return &(iter->second);
    }

    iter->second = d;
    return &(iter->second);
  }

  DATA* insert(const KEY& k, const DATA& d)
  {
    return push(k, d);
  }

  void pop()
  {
    data.pop_back();
  }

  const std::pair<KEY, DATA>& top() const
  {
    return data[size() - 1];
  }

  DATA* findPtr(const KEY& k)
  {
    iterator iter = find(k);
    if (iter == data.end())
      return nullptr;
    return &(iter->second);
  }

  const_iterator find(const KEY& k) const
  {
    const_iterator iter = std::lower_bound(data.begin(), data.end(), k, compare);
    return iter != data.end() && !compare(k, *iter) ? iter : data.end();
  }

  iterator find(const KEY& k)
  {
    iterator iter = std::lower_bound(data.begin(), data.end(), k, compare);
    return iter != data.end() && !compare(k, *iter) ? iter : data.end();
  }

  DATA& operator[](const KEY& k)
  {
    iterator iter = std::lower_bound(data.begin(), data.end(), k, compare);
    if (iter == data.end() || compare(k, *iter))
    {
      DATA d = DATA();
      iter = data.insert(iter, std::make_pair(k, d));
    }
    return iter->second;
  }

  const std::pair<KEY, DATA>& getAt(size_t index)
  {
    return data[index];
  }

  bool empty() const
  {
    return data.empty();
  }

  size_t size() const
  {
    return data.size();
  }

  void clear()
  {
    return data.clear();
  }

  const_iterator begin() const
  {
    return data.begin();
  }

  const_iterator end() const
  {
    return data.end();
  }

  iterator begin()
  {
    return data.begin();
  }

  iterator end()
  {
    return data.end();
  }

  ~OrderedVector()
  {
    data.clear();
  }

protected:
  class PairCompare
  {
  public:
    bool operator()(const KEY& l, const KEY& r) const
    {
      return kOrderRel(l, r);
    }

    bool operator()(const std::pair<KEY, DATA>& l, const std::pair<KEY, DATA>& r) const
    {
      return kOrderRel(l.first, r.first);
    }

    bool operator()(const KEY& l, const std::pair<KEY, DATA>& r) const
    {
      return kOrderRel(l, r.first);
    }

    bool operator()(const std::pair<KEY, DATA>& l, const KEY& r) const
    {
      return kOrderRel(l.first, r);
    }

    KEY_ORDER_REL kOrderRel;
  };

  std::vector<std::pair<KEY, DATA>> data;
  PairCompare compare;
};

