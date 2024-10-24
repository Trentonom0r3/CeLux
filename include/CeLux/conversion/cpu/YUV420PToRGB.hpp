// YUV420PToRGB.hpp
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
 * @brief Converter for NV12 to RGB conversion on CPU.
 *
 * @tparam T Data type used for conversion.
 */
class YUV420PToRGB : public ConverterBase
{
  public:
    /**
     * @brief Constructor that invokes the base class constructor.
     */
    YUV420PToRGB() : ConverterBase()
    {
    }

    /**
     * @brief Destructor that frees the swsContext.
     */
    ~YUV420PToRGB()
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
    void convert(celux::Frame& frame, void* buffer) override
    {
        // Verify the pixel format
        if (frame.getPixelFormat() != AV_PIX_FMT_YUV420P)
        {
            std::cout << "Format not YUV420. Format is actually: "
                      << av_get_pix_fmt_name(frame.getPixelFormat()) << std::endl;
        }

        if (!swsContext)
        {
            // Initialize the swsContext for YUV420P to RGB conversion
            swsContext = sws_getContext(frame.getWidth(), frame.getHeight(),
                                        AV_PIX_FMT_YUV420P, // Source format
                                        frame.getWidth(), frame.getHeight(),
                                        AV_PIX_FMT_RGB24, // Destination format
                                        SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (!swsContext)
            {
                throw std::runtime_error(
                    "Failed to initialize swsContext for YUV420P to RGB conversion");
            }

            // Set color space and range explicitly (optional)
            int srcRange = 0; // MPEG (16-235)
            int dstRange = 1; // JPEG (0-255)
            const int* srcMatrix = sws_getCoefficients(SWS_CS_ITU709);
            const int* dstMatrix = sws_getCoefficients(SWS_CS_ITU709);

            sws_setColorspaceDetails(swsContext, srcMatrix, srcRange, dstMatrix,
                                     dstRange, 0, 1 << 16, 1 << 16);
        }

        // Source data and line sizes
        const uint8_t* srcData[4] = {nullptr};
        int srcLineSize[4] = {0};

        srcData[0] = frame.getData(0);
        srcData[1] = frame.getData(1);
        srcData[2] = frame.getData(2);

        srcLineSize[0] = frame.getLineSize(0);
        srcLineSize[1] = frame.getLineSize(1);
        srcLineSize[2] = frame.getLineSize(2);

        // Destination data and line sizes
        uint8_t* dstData[4] = {nullptr};
        int dstLineSize[4] = {0};

        // Calculate the required buffer size
        int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, frame.getWidth(),
                                                frame.getHeight(), 1);
        if (numBytes < 0)
        {
            throw std::runtime_error("Could not get buffer size");
        }

        // Initialize the destination data pointers and line sizes
        int ret = av_image_fill_arrays(dstData, dstLineSize,
                                       static_cast<uint8_t*>(buffer), AV_PIX_FMT_RGB24,
                                       frame.getWidth(), frame.getHeight(), 1);
        if (ret < 0)
        {
            throw std::runtime_error("Could not fill destination image arrays");
        }

        // Perform the conversion from YUV420P to RGB
        int result = sws_scale(swsContext, srcData, srcLineSize, 0, frame.getHeight(),
                               dstData, dstLineSize);
        if (result <= 0)
        {
            throw std::runtime_error("sws_scale failed during conversion");
        }
    }
};

} // namespace cpu
} // namespace conversion
} // namespace celux
