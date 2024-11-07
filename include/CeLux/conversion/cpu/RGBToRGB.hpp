// RGBToRGB.hpp
#pragma once

#include "CPUConverter.hpp"

namespace celux
{
namespace conversion
{
namespace cpu
{

/**
 * @brief Converter for NV12 to RGB conversion on CPU.
 *
 * @tparam T Data type used for conversion.
 */
class RGBToRGB : public ConverterBase
{
  public:
    /**
     * @brief Constructor that invokes the base class constructor.
     */
    RGBToRGB() : ConverterBase()
    {
    }

    /**
     * @brief Destructor that frees the swsContext.
     */
    ~RGBToRGB()
    {
        if (swsContext)
        {
            sws_freeContext(swsContext);
            swsContext = nullptr;
        }
    }

    /**
     * @brief Performs BGR to RGB conversion.
     *
     * @param frame Reference to the frame to be converted.
     * @param buffer Pointer to the buffer that will store the converted frame.
     */
    void convert(celux::Frame& frame, void* buffer) override
    {

		memcpy(buffer, frame.get()->data[0],
               frame.get()->linesize[0] * frame.get()->height);

    }
};

} // namespace cpu
} // namespace conversion
} // namespace celux
