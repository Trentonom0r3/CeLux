// RGBToNV12.hpp
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
 * @brief Converter for RGB to NV12 conversion on CPU.
 *
 * @tparam T Data type used for conversion.
 */
template <typename T> class RGBToNV12 : public ConverterBase<T>
{
  public:
    /**
     * @brief Constructor that invokes the base class constructor.
     */
    RGBToNV12() : ConverterBase<T>()
    {
    }

    /**
     * @brief Destructor that frees the swsContext.
     */
    ~RGBToNV12()
    {
        if (swsContext)
        {
            sws_freeContext(swsContext);
            swsContext = nullptr;
        }
    }

    /**
     * @brief Performs RGB to NV12 conversion.
     *
     * @param frame Reference to the frame to be converted.
     * @param buffer Pointer to the buffer containing NV12 data.
     */
    void convert(celux::Frame& frame, void* sourceBuffer) override
    {
        CELUX_DEBUG("RGBToNV12::convert()");
        CELUX_DEBUG("Frame FORMAT: {}", frame.getPixelFormatString());
        if (!swsContext)
        {
            // Initialize the swsContext for RGB to NV12 conversion
            swsContext = sws_getContext(frame.getWidth(), frame.getHeight(),
                                        AV_PIX_FMT_RGB24, // Source format
                                        frame.getWidth(), frame.getHeight(),
                                        AV_PIX_FMT_NV12, // Destination format
                                        SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (!swsContext)
            {
                throw std::runtime_error(
                    "Failed to initialize swsContext for RGB to NV12 conversion");
            }

            // Set color space and range explicitly (optional)
            int srcRange = 1; // JPEG (0-255)
            int dstRange = 0; // MPEG (16-235)
            const int* srcMatrix = sws_getCoefficients(SWS_CS_ITU709);
            const int* dstMatrix = sws_getCoefficients(SWS_CS_ITU709);

            sws_setColorspaceDetails(swsContext, srcMatrix, srcRange, dstMatrix,
                                     dstRange, 0, 1 << 16, 1 << 16);
        }

        // Source data and line sizes
        const uint8_t* srcData[4] = {static_cast<uint8_t*>(sourceBuffer), nullptr,
                                     nullptr, nullptr};
        int srcLineSize[4] = {frame.getWidth() * 3, 0, 0,
                              0}; // RGB24 has 3 bytes per pixel

        // Destination data and line sizes are from the AVFrame
        uint8_t* dstData[4] = {frame.getData(0), frame.getData(1), nullptr, nullptr};
        int dstLineSize[4] = {frame.getLineSize(0), frame.getLineSize(1), 0, 0};

        // Perform the conversion from RGB to NV12
        int result = sws_scale(swsContext, srcData, srcLineSize, 0, frame.getHeight(),
                               dstData, dstLineSize);
        if (result <= 0)
        {
            throw std::runtime_error("sws_scale failed during conversion");
        }
    }


  private:
    struct SwsContext* swsContext = nullptr;
};

} // namespace cpu
} // namespace conversion
} // namespace celux
