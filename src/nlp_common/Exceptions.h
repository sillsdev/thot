#pragma once

#include <stdexcept>

class NotImplemented : public std::logic_error
{
public:
  NotImplemented() : std::logic_error("Not implemented")
  {
  }
};
