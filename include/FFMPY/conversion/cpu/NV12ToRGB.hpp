// NV12ToRGB.hpp
#pragma once

#include "CPUConverter.hpp"
#include "Frame.hpp"
#include <type_traits>

namespace ffmpy
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
template <typename T> class NV12ToRGB : public ConverterBase<T>
{
  public:
    /**
     * @brief Constructor that invokes the base class constructor.
     */
    NV12ToRGB() : ConverterBase<T>()
    {
    }

    /**
     * @brief Destructor that frees the swsContext.
     */
    ~NV12ToRGB()
    {
        if (swsContext)
        {
            sws_freeContext(swsContext);
            swsContext = nullptr;
        }
    }

    /**
     * @brief Performs NV12 to RGB conversion.
     *
     * @param frame Reference to the frame to be converted.
     * @param buffer Pointer to the buffer containing NV12 data.
     */
    void convert(ffmpy::Frame& frame, void* buffer) override
    {
        if (!swsContext)
        {
            // Initialize the swsContext for NV12 to RGB conversion
            swsContext = sws_getContext(frame.getWidth(), frame.getHeight(),
                                        AV_PIX_FMT_NV12, // Source format
                                        frame.getWidth(), frame.getHeight(),
                                        AV_PIX_FMT_RGB24, // Destination format
                                        SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (!swsContext)
            {
                throw std::runtime_error(
                    "Failed to initialize swsContext for NV12 to RGB conversion");
            }
        }

        const uint8_t* srcData[2] = {frame.getData(0), frame.getData(1)};
        int srcLineSize[2] = {frame.getLineSize(0), frame.getLineSize(1)};

        uint8_t* dstData[1] = {static_cast<uint8_t*>(buffer)};
        int dstLineSize[1] = {frame.getWidth() * 3}; // RGB stride

        // Perform the conversion from NV12 to RGB
        sws_scale(swsContext, srcData, srcLineSize, 0, frame.getHeight(), dstData,
                  dstLineSize);
    }
};

} // namespace cpu
} // namespace conversion
} // namespace ffmpy
