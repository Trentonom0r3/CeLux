#pragma once

#include "CPUConverter.hpp"

namespace celux
{
namespace conversion
{
namespace cpu
{

/**
 * @brief Converter for RGB24 to YUV420P conversion on CPU.
 */
class RGBToYUV420P : public ConverterBase
{
  public:
    /**
     * @brief Constructor initializes the converter.
     */
    RGBToYUV420P() : ConverterBase()
    {
    }

    /**
     * @brief Destructor frees the swsContext.
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
     * @brief Converts an RGB24 frame to YUV420P.
     *
     * @param frame Reference to the output frame (YUV420P).
     * @param buffer Pointer to the input buffer (RGB24).
     */
    void convert(celux::Frame& frame, void* buffer) override
    {
        // Ensure the frame is YUV420P
        if (frame.getPixelFormat() != AV_PIX_FMT_YUV420P)
        {
            std::cerr << "Frame format is not YUV420P. Actual format: "
                      << av_get_pix_fmt_name(frame.getPixelFormat()) << std::endl;
            throw std::runtime_error("Frame format mismatch: Expected YUV420P");
        }

        if (!swsContext)
        {
            // Initialize the swsContext for RGB24 to YUV420P conversion
            swsContext = sws_getContext(frame.getWidth(), frame.getHeight(),
                                        AV_PIX_FMT_RGB24, // Source format
                                        frame.getWidth(), frame.getHeight(),
                                        AV_PIX_FMT_YUV420P, // Destination format
                                        SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (!swsContext)
            {
                throw std::runtime_error(
                    "Failed to initialize swsContext for RGB24 to YUV420P conversion");
            }

            // Set color conversion details (default settings)
            int srcRange = 1; // Full range (0-255)
            int dstRange = 0; // Limited range (16-235)
            const int* srcMatrix = sws_getCoefficients(SWS_CS_DEFAULT);
            const int* dstMatrix = sws_getCoefficients(SWS_CS_DEFAULT);

            sws_setColorspaceDetails(swsContext, srcMatrix, srcRange, dstMatrix,
                                     dstRange, 0, 1 << 16, 1 << 16);
        }

        // Source data and line sizes (RGB24)
        const uint8_t* srcData[4] = {static_cast<uint8_t*>(buffer), nullptr, nullptr,
                                     nullptr};
        int srcLineSize[4] = {frame.getWidth() * 3, 0, 0,
                              0}; // RGB24: 3 bytes per pixel

        // Destination data and line sizes (YUV420P)
        uint8_t* dstData[4] = {nullptr};
        int dstLineSize[4] = {0};

        dstData[0] = frame.getData(0); // Y plane
        dstData[1] = frame.getData(1); // U plane
        dstData[2] = frame.getData(2); // V plane

        dstLineSize[0] = frame.getLineSize(0);
        dstLineSize[1] = frame.getLineSize(1);
        dstLineSize[2] = frame.getLineSize(2);

        // Perform the RGB24 to YUV420P conversion
        int result = sws_scale(swsContext, srcData, srcLineSize, 0, frame.getHeight(),
                               dstData, dstLineSize);
        if (result <= 0)
        {
            throw std::runtime_error(
                "sws_scale failed during RGB to YUV420P conversion");
        }
    }
};

} // namespace cpu
} // namespace conversion
} // namespace celux
