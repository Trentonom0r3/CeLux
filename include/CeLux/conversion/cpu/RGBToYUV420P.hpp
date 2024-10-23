// RGBToYUV420P.hpp
#pragma once

#include "CPUConverter.hpp"
#include "Frame.hpp"
#include <type_traits>

namespace celux
{
namespace conversion
{
namespace cpu
{

/**
 * @brief Converter for RGB to YUV420P conversion on CPU.
 *
 * @tparam T Data type used for conversion.
 */
class RGBToYUV420P : public ConverterBase
{
  public:
    /**
     * @brief Constructor that invokes the base class constructor.
     */
    RGBToYUV420P() : ConverterBase()
    {
    }

    /**
     * @brief Destructor that frees the swsContext.
     */
    ~RGBToYUV420P()
    {
        if (swsContext)
        {
            sws_freeContext(swsContext);
            swsContext = nullptr;
        }
    }

    /**
     * @brief Performs RGB to YUV420P conversion.
     *
     * @param frame Reference to the frame to be converted.
     * @param sourceBuffer Pointer to the buffer containing RGB24 data.
     */
    void convert(celux::Frame& frame, void* sourceBuffer) override
    {
        CELUX_DEBUG("RGBToYUV420P::convert()");
        CELUX_DEBUG("Frame FORMAT: {}", frame.getPixelFormatString());

        if (!swsContext)
        {
            // Initialize the swsContext for RGB to YUV420P conversion
            swsContext = sws_getContext(frame.getWidth(), frame.getHeight(),
                                        AV_PIX_FMT_RGB24, // Source format
                                        frame.getWidth(), frame.getHeight(),
                                        AV_PIX_FMT_YUV420P, // Destination format
                                        SWS_BILINEAR, nullptr, nullptr, nullptr);

            if (!swsContext)
            {
                throw std::runtime_error(
                    "Failed to initialize swsContext for RGB to YUV420P conversion");
            }

            // Set colorspace and range explicitly (optional)
            // Example: ITU-R BT.601 colorspace
            sws_setColorspaceDetails(swsContext, sws_getCoefficients(SWS_CS_ITU709),
                                     SWS_CS_ITU709, sws_getCoefficients(SWS_CS_ITU709),
                                     SWS_CS_ITU709, 0, 1 << 16, 1 << 16);
        }

        // Source data and line sizes
        const uint8_t* srcData[4] = {static_cast<uint8_t*>(sourceBuffer), nullptr,
                                     nullptr, nullptr};
        int srcLineSize[4] = {frame.getWidth() * 3, 0, 0,
                              0}; // RGB24 has 3 bytes per pixel

        // Destination data and line sizes for YUV420P
        uint8_t* dstData[4] = {frame.getData(0), // Y plane
                               frame.getData(1), // U plane
                               frame.getData(2), // V plane
                               nullptr};
        int dstLineSize[4] = {frame.getLineSize(0), // Y plane
                              frame.getLineSize(1), // U plane
                              frame.getLineSize(2), // V plane
                              0};

        // Perform the conversion from RGB to YUV420P
        int result = sws_scale(swsContext, srcData, srcLineSize, 0, frame.getHeight(),
                               dstData, dstLineSize);
        if (result <= 0)
        {
            throw std::runtime_error(
                "sws_scale failed during RGB to YUV420P conversion");
        }
    }

  private:
    SwsContext* swsContext = nullptr;
};

} // namespace cpu
} // namespace conversion
} // namespace celux
