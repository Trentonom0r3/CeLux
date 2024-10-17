// IConverter.hpp

#pragma once

#include "Frame.hpp"

namespace celux
{
namespace conversion
{

class IConverter
{
  public:
    virtual ~IConverter()
    {
    }
    virtual void convert(celux::Frame& frame, void* buffer) = 0;
    virtual void synchronize() = 0;
};

} // namespace conversion
} // namespace celux
