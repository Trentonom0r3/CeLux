// RGB48ToYUV420P10.hpp
#pragma once

#include "CPUConverter.hpp"
#include "Frame.hpp"
#include <cstring> // For memcpy
#include <iostream>
#include <stdexcept>
#include <type_traits>

extern "C"
{
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

namespace celux
{
namespace conversion
{
namespace cpu
{

/**
 * @brief Converter for RGB48LE to YUV420P10LE conversion on CPU.
 *
 * This converter handles 48-bit RGB (RGB48LE) input and converts it to
 * 10-bit YUV420 planar (YUV420P10LE) output using FFmpeg's sws_scale.
 */
class RGB48ToYUV420P10 : public ConverterBase
{
  public:
    /**
     * @brief Constructor that initializes the base class and swsContext to nullptr.
     */
    RGB48ToYUV420P10() : ConverterBase(), swsContext(nullptr)
    {
    }

    /**
     * @brief Destructor that frees the swsContext if it was initialized.
     */
    ~RGB48ToYUV420P10()
    {
        if (swsContext)
        {
            sws_freeContext(swsContext);
            swsContext = nullptr;
        }
    }

    /**
     * @brief Performs RGB48LE to YUV420P10LE conversion.
     *
     * @param frame Reference to the frame to be converted.
     * @param buffer Pointer to the buffer where YUV420P10LE data will be stored.
     *
     * @throws std::runtime_error if conversion fails or unsupported formats are
     * provided.
     */
    void convert(celux::Frame& frame, void* buffer) override
    {
        try
        {
            CELUX_DEBUG("ATTEMPTING RGB48LE to YUV420P10LE CONVERSION");

            // Verify the destination frame's pixel format
            if (frame.getPixelFormat() != AV_PIX_FMT_YUV420P10LE)
            {
                std::cerr << "Destination frame format is not YUV420P10LE. It is: "
                          << av_get_pix_fmt_name(frame.getPixelFormat()) << std::endl;
                throw std::invalid_argument(
                    "Unsupported pixel format for YUV420P10LE conversion.");
            }

            CELUX_DEBUG("Starting RGB48LE to YUV420P10LE conversion.");

            // Initialize the swsContext if not already done
            if (!swsContext)
            {
                swsContext =
                    sws_getContext(frame.getWidth(), frame.getHeight(),
                                   AV_PIX_FMT_RGB48LE, // Source format
                                   frame.getWidth(), frame.getHeight(),
                                   AV_PIX_FMT_YUV420P10LE, // Destination format
                                   SWS_BILINEAR, nullptr, nullptr, nullptr);

                CELUX_DEBUG(
                    "SWS Context initialized for RGB48LE to YUV420P10LE conversion.");
                if (!swsContext)
                {
                    throw std::runtime_error("Failed to initialize swsContext for "
                                             "RGB48LE to YUV420P10LE conversion.");
                }

                // Optionally, set color space and range
                int srcRange = 1; // JPEG (0-255)
                int dstRange = 0; // MPEG (16-235)
                const int* srcMatrix = sws_getCoefficients(SWS_CS_ITU709);
                const int* dstMatrix = sws_getCoefficients(SWS_CS_ITU709);
                CELUX_DEBUG("Setting colorspace details.");
                sws_setColorspaceDetails(swsContext, srcMatrix, srcRange, dstMatrix,
                                         dstRange, 0, 1 << 16, 1 << 16);
            }

            // Source data and line sizes (RGB48LE from buffer)
            const uint8_t* srcData[4] = {nullptr};
            int srcLineSize[4] = {0};

            // Assuming buffer is in HWC format (Height-Width-Channel)
            // and tightly packed (no padding), calculate pointers accordingly
            // Each channel is 16 bits (uint16_t), so stride is width * 3 *
            // sizeof(uint16_t)
            uint16_t* rgbBuffer = static_cast<uint16_t*>(buffer);
            int width = frame.getWidth();
            int height = frame.getHeight();

            // For AV_PIX_FMT_RGB48LE, FFmpeg expects interleaved data:
            // R1 R2 ... G1 G2 ... B1 B2 ... per pixel
            // Therefore, srcData[0] points to the start of the buffer
            // and srcLineSize[0] is the number of bytes per row

            srcData[0] = reinterpret_cast<uint8_t*>(rgbBuffer);
            srcLineSize[0] = width * 3 * sizeof(uint16_t); // 3 channels, 16 bits each

            // Destination data and line sizes (YUV420P10LE in frame)
            uint8_t* dstData[4] = {nullptr};
            int dstLineSize[4] = {0};

            // Initialize destination data pointers from frame
            for (int i = 0; i < 3; ++i)
            { // Y, U, V planes
                dstData[i] = frame.getData(i);
                dstLineSize[i] = frame.getLineSize(i);
            }

            // Perform the conversion from RGB48LE to YUV420P10LE
            CELUX_DEBUG("Starting sws_scale.");
            int result = sws_scale(swsContext, srcData, srcLineSize, 0, height, dstData,
                                   dstLineSize);

            CELUX_DEBUG("sws_scale result: {}", result);

            if (result <= 0)
            {
                throw std::runtime_error(
                    "sws_scale failed during RGB48LE to YUV420P10LE conversion.");
            }

            CELUX_DEBUG("Conversion successful.");
        }
        catch (const std::exception& e)
        {
            CELUX_DEBUG("Error in RGB48LE to YUV420P10LE conversion: {}", e.what());
            throw; // Re-throw exception after logging
        }
    }


  private:
    struct SwsContext* swsContext;
};

} // namespace cpu
} // namespace conversion
} // namespace celux
