// ConverterBase.hpp

#pragma once
#include "Frame.hpp"

namespace ffmpy
{
namespace conversion
{
namespace cpu
{
template <typename T> class ConverterBase : public IConverter
{
  public:
    ConverterBase();
    virtual ~ConverterBase();
    virtual void convert(ffmpy::Frame& frame,
                         void* buffer) = 0; /// to be implemented by derived classes
    virtual void synchronize(){};
  protected:
    // ffmpy conversion contexts
};

} // namespace cpu
} // namespace conversion
} // namespace ffmpy
