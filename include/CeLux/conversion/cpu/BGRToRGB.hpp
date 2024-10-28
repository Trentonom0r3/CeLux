// BGRToRGB.hpp
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
class BGRToRGB : public ConverterBase
{
  public:
    /**
     * @brief Constructor that invokes the base class constructor.
     */
    BGRToRGB() : ConverterBase()
    {
    }

    /**
     * @brief Destructor that frees the swsContext.
     */
    ~BGRToRGB()
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
        if (!swsContext)
        {
            swsContext = sws_getContext(frame.getWidth(), frame.getHeight(),
                										AV_PIX_FMT_BGR24, // Source format
                										frame.getWidth(), frame.getHeight(),
                										AV_PIX_FMT_RGB24, // Destination format
                										SWS_BILINEAR, nullptr, nullptr, nullptr);
		}

		AVFrame* avFrame = frame.get();
		AVFrame* outFrame = av_frame_alloc();
        if (!outFrame)
        {
			throw std::runtime_error("Failed to allocate AVFrame");
		}

		outFrame->format = AV_PIX_FMT_RGB24;
		outFrame->width = avFrame->width;
		outFrame->height = avFrame->height;

		int ret = av_frame_get_buffer(outFrame, 32);
        if (ret < 0)
        {
			throw std::runtime_error("Failed to allocate frame buffer");
		}

		ret = sws_scale(swsContext, avFrame->data, avFrame->linesize, 0, avFrame->height,
            						outFrame->data, outFrame->linesize);
        if (ret < 0)
        {
			throw std::runtime_error("Failed to scale frame");
		}

		std::memcpy(buffer, outFrame->data[0], outFrame->linesize[0] * outFrame->height);

		av_frame_free(&outFrame);

    }
};

} // namespace cpu
} // namespace conversion
} // namespace celux
