// IConverter.hpp

#pragma once

#include "Frame.hpp"
#include <cuda_runtime.h>

namespace ffmpy
{
namespace conversion
{

class IConverter
{
  public:
    virtual ~IConverter()
    {
    }
    virtual void convert(ffmpy::Frame& frame, void* buffer) = 0;
    virtual void synchronize() = 0;
};

} // namespace conversion
} // namespace ffmpy
