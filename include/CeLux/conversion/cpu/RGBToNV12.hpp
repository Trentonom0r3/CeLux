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
    void convert(celux::Frame& frame, void* buffer) override
    {
        CELUX_DEBUG("RGBToNV12::convert()");
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
        const uint8_t* srcData[4] = {nullptr};
        int srcLineSize[4] = {0};

        srcData[0] = frame.getData(0);
        srcLineSize[0] = frame.getLineSize(0);

        // Destination data and line sizes
        uint8_t* dstData[4] = {nullptr};
        int dstLineSize[4] = {0};

        // Calculate the required buffer size
        int numBytes = av_image_get_buffer_size(AV_PIX_FMT_NV12, frame.getWidth(),
                                                frame.getHeight(), 1);
        if (numBytes < 0)
        {
            throw std::runtime_error("Could not get buffer size");
        }

        // Initialize the destination data pointers and line sizes
        int ret = av_image_fill_arrays(dstData, dstLineSize,
                                       static_cast<uint8_t*>(buffer), AV_PIX_FMT_NV12,
                                       frame.getWidth(), frame.getHeight(), 1);
        if (ret < 0)
        {
            throw std::runtime_error("Could not fill destination image arrays");
        }

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
