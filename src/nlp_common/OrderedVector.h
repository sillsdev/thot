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

#ifndef _OrderedVector_h
#define _OrderedVector_h

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

  OrderedVector();
  OrderedVector(const OrderedVector<KEY, DATA, KEY_ORDER_REL>& ov);
  // Copy-constructor
  OrderedVector<KEY, DATA, KEY_ORDER_REL>& operator=(const OrderedVector<KEY, DATA, KEY_ORDER_REL>& ov);
  DATA* push(const KEY& k, const DATA& d);
  DATA* insert(const KEY& k, const DATA& d);
  void pop();
  const std::pair<KEY, DATA>& top() const;
  DATA* findPtr(const KEY& k);
  iterator find(const KEY& k);
  DATA& operator[](const KEY& k);
  const std::pair<KEY, DATA>& getAt(size_t index);
  bool empty() const;
  size_t size() const;
  void clear();

  // Constant iterator functions for the OrderedVector class
  const_iterator begin() const;
  const_iterator end() const;

  // Iterator functions for the OrderedVector class
  iterator begin();
  iterator end();

  ~OrderedVector();

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

template <class KEY, class DATA, class KEY_ORDER_REL>
OrderedVector<KEY, DATA, KEY_ORDER_REL>::OrderedVector()
{
}

template <class KEY, class DATA, class KEY_ORDER_REL>
OrderedVector<KEY, DATA, KEY_ORDER_REL>::OrderedVector(const OrderedVector<KEY, DATA, KEY_ORDER_REL>& ov)
    : data(ov.data)
{
}

template <class KEY, class DATA, class KEY_ORDER_REL>
OrderedVector<KEY, DATA, KEY_ORDER_REL>& OrderedVector<KEY, DATA, KEY_ORDER_REL>::operator=(
    const OrderedVector<KEY, DATA, KEY_ORDER_REL>& ov)
{
  if (&ov != this)
    data = ov.data;
  return *this;
}

template <class KEY, class DATA, class KEY_ORDER_REL>
DATA* OrderedVector<KEY, DATA, KEY_ORDER_REL>::push(const KEY& k, const DATA& d)
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

template <class KEY, class DATA, class KEY_ORDER_REL>
DATA* OrderedVector<KEY, DATA, KEY_ORDER_REL>::insert(const KEY& k, const DATA& d)
{
  return push(k, d);
}

template <class KEY, class DATA, class KEY_ORDER_REL>
void OrderedVector<KEY, DATA, KEY_ORDER_REL>::pop()
{
  data.pop_back();
}

template <class KEY, class DATA, class KEY_ORDER_REL>
const std::pair<KEY, DATA>& OrderedVector<KEY, DATA, KEY_ORDER_REL>::top() const
{
  return data[size() - 1];
}

template <class KEY, class DATA, class KEY_ORDER_REL>
DATA* OrderedVector<KEY, DATA, KEY_ORDER_REL>::findPtr(const KEY& k)
{
  iterator iter = find(k);
  if (iter == data.end())
    return nullptr;
  return &(iter->second);
}

template <class KEY, class DATA, class KEY_ORDER_REL>
typename OrderedVector<KEY, DATA, KEY_ORDER_REL>::iterator OrderedVector<KEY, DATA, KEY_ORDER_REL>::find(const KEY& k)
{
  iterator iter = std::lower_bound(data.begin(), data.end(), k, compare);
  return iter != data.end() && !compare(k, *iter) ? iter : data.end();
}

template <class KEY, class DATA, class KEY_ORDER_REL>
DATA& OrderedVector<KEY, DATA, KEY_ORDER_REL>::operator[](const KEY& k)
{
  iterator iter = std::lower_bound(data.begin(), data.end(), k, compare);
  if (iter == data.end() || compare(k, *iter))
  {
    DATA d = DATA();
    iter = data.insert(iter, std::make_pair(k, d));
  }
  return iter->second;
}

template <class KEY, class DATA, class KEY_ORDER_REL>
const std::pair<KEY, DATA>& OrderedVector<KEY, DATA, KEY_ORDER_REL>::getAt(size_t index)
{
  return data[index];
}

template <class KEY, class DATA, class KEY_ORDER_REL>
bool OrderedVector<KEY, DATA, KEY_ORDER_REL>::empty() const
{
  return this->size() == 0;
}

template <class KEY, class DATA, class KEY_ORDER_REL>
size_t OrderedVector<KEY, DATA, KEY_ORDER_REL>::size() const
{
  return data.size();
}

template <class KEY, class DATA, class KEY_ORDER_REL>
void OrderedVector<KEY, DATA, KEY_ORDER_REL>::clear()
{
  data.clear();
}

template <class KEY, class DATA, class KEY_ORDER_REL>
typename OrderedVector<KEY, DATA, KEY_ORDER_REL>::const_iterator OrderedVector<KEY, DATA, KEY_ORDER_REL>::begin() const
{
  return data.begin();
}

template <class KEY, class DATA, class KEY_ORDER_REL>
typename OrderedVector<KEY, DATA, KEY_ORDER_REL>::const_iterator OrderedVector<KEY, DATA, KEY_ORDER_REL>::end() const
{
  return data.end();
}

template <class KEY, class DATA, class KEY_ORDER_REL>
typename OrderedVector<KEY, DATA, KEY_ORDER_REL>::iterator OrderedVector<KEY, DATA, KEY_ORDER_REL>::begin()
{
  return data.begin();
}

template <class KEY, class DATA, class KEY_ORDER_REL>
typename OrderedVector<KEY, DATA, KEY_ORDER_REL>::iterator OrderedVector<KEY, DATA, KEY_ORDER_REL>::end()
{
  return data.end();
}

template <class KEY, class DATA, class KEY_ORDER_REL>
OrderedVector<KEY, DATA, KEY_ORDER_REL>::~OrderedVector()
{
  clear();
}

#endif
