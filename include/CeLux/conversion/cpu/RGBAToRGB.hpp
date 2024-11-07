// RGBAToRGB.hpp
#pragma once

#include "CPUConverter.hpp"


namespace celux
{
namespace conversion
{
namespace cpu
{

/**
 * @brief Converter for RGBA to RGB24 conversion on CPU.
 */
class RGBAToRGB : public ConverterBase
{
  public:
    /**
     * @brief Constructor that initializes the swsContext.
     */
    RGBAToRGB() : ConverterBase(), swsContext(nullptr)
    {
        CELUX_DEBUG("Initializing RGBAToRGB converter");
    }

    /**
     * @brief Destructor that frees the swsContext.
     */
    ~RGBAToRGB()
    {
        if (swsContext)
        {
            sws_freeContext(swsContext);
            swsContext = nullptr;
        }
    }

    /**
     * @brief Performs RGBA to RGB24 conversion.
     *
     * @param frame Reference to the frame to be converted.
     * @param buffer Pointer to the buffer where RGB24 data will be stored.
     *
     * @throws std::runtime_error if conversion fails or unsupported formats are
     * provided.
     */
    void convert(celux::Frame& frame, void* buffer) override
    {
        // Verify the pixel format
        if (frame.getPixelFormat() != AV_PIX_FMT_RGBA)
        {
            std::cerr << "Format not RGBA. Format is actually: "
                      << av_get_pix_fmt_name(frame.getPixelFormat()) << std::endl;
            throw std::invalid_argument(
                "Unsupported pixel format for RGBAToRGB converter.");
        }

        // Initialize the swsContext if not already done
        if (!swsContext)
        {
            swsContext = sws_getContext(frame.getWidth(), frame.getHeight(),
                                        AV_PIX_FMT_RGBA, // Source format
                                        frame.getWidth(), frame.getHeight(),
                                        AV_PIX_FMT_RGB24, // Destination format
                                        SWS_BILINEAR, nullptr, nullptr, nullptr);

            if (!swsContext)
            {
                throw std::runtime_error(
                    "Failed to initialize swsContext for RGBA to RGB24 conversion");
            }

            // Optionally set colorspace details
            sws_setColorspaceDetails(swsContext, sws_getCoefficients(SWS_CS_DEFAULT), 0,
                                     sws_getCoefficients(SWS_CS_DEFAULT), 1, 0, 1 << 16,
                                     1 << 16);
        }

        // Source data and line sizes
        const uint8_t* srcData[4] = {nullptr};
        int srcLineSize[4] = {0};
        srcData[0] = frame.getData(0); // RGBA data
        srcLineSize[0] = frame.getLineSize(0);

        // Destination data and line sizes
        uint8_t* dstData[4] = {nullptr};
        int dstLineSize[4] = {0};

        // Calculate the required buffer size for RGB24
        int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, frame.getWidth(),
                                                frame.getHeight(), 1);

        if (numBytes < 0)
        {
            throw std::runtime_error("Could not get buffer size for RGB24");
        }

        // Initialize the destination data pointers and line sizes to point to the user
        // buffer
        int ret = av_image_fill_arrays(dstData, dstLineSize,
                                       static_cast<uint8_t*>(buffer), AV_PIX_FMT_RGB24,
                                       frame.getWidth(), frame.getHeight(), 1);

        if (ret < 0)
        {
            char errBuf[256];
            av_strerror(ret, errBuf, sizeof(errBuf));
            throw std::runtime_error(std::string("av_image_fill_arrays failed: ") +
                                     errBuf);
        }

        // Perform the conversion from RGBA to RGB24 directly into user buffer
        int result = sws_scale(swsContext, srcData, srcLineSize, 0, frame.getHeight(),
                               dstData, dstLineSize);

        if (result <= 0)
        {
            throw std::runtime_error(
                "sws_scale failed during RGBA to RGB24 conversion");
        }
    }

  private:
    struct SwsContext* swsContext;
};

} // namespace cpu
} // namespace conversion
} // namespace celux
