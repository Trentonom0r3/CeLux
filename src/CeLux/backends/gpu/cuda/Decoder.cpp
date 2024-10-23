// CUDA Decoder.cpp
#include "backends/gpu/cuda/Decoder.hpp"
// include cuda streams
using namespace celux::error;

namespace celux::backends::gpu::cuda
{

Decoder::~Decoder()
{
    CELUX_DEBUG("CUDA Decoder destructor called");
}

void Decoder::initCodecContext(const AVCodec* codec)
{
    CELUX_INFO("Initializing codec context for CUDA Decoder");
    // Call base class implementation
    CELUX_DEBUG("Calling base class Decoder::initCodecContext");
    celux::Decoder::initCodecContext(codec);
    CELUX_DEBUG("Base class Decoder::initCodecContext completed");

    // Set hardware-specific get_format
    if (hwDeviceCtx)
    {
        CELUX_DEBUG("Setting hardware-specific get_format to Decoder::getHWFormat");
        codecCtx->get_format = Decoder::getHWFormat; // Assign the static function
        codecCtx->sw_pix_fmt = AV_PIX_FMT_YUV420P;    // Set the software pixel format
    }
    else
    {
        CELUX_WARN("Hardware device context is not initialized; get_format not set");
    }
    CELUX_INFO("Codec context initialization for CUDA Decoder completed");
}

void Decoder::initHWAccel()
{
    CELUX_INFO("Initializing hardware acceleration for CUDA Decoder");

    // Find the hardware device type
    CELUX_DEBUG("Searching for HW device type 'cuda'");
    enum AVHWDeviceType type = av_hwdevice_find_type_by_name("cuda");
    if (type == AV_HWDEVICE_TYPE_NONE)
    {
        CELUX_ERROR("Failed to find HW device type: cuda");
        throw CxException("Failed to find HW device type: cuda");
    }
    CELUX_DEBUG("HW device type 'cuda' found: {}", av_hwdevice_get_type_name(type));

    // Initialize hardware device context
    CELUX_DEBUG("Creating HW device context for type: {}",
                av_hwdevice_get_type_name(type));
    AVBufferRef* hw_ctx = nullptr;
    int ret = av_hwdevice_ctx_create(&hw_ctx, type, nullptr, nullptr, 0);
    if (ret < 0)
    {
        CELUX_ERROR("Failed to create HW device context for 'cuda': error code {}",
                    ret);
        FF_CHECK_MSG(ret, std::string("Failed to create HW device context:"));
    }
    hwDeviceCtx.reset(hw_ctx);
    CELUX_INFO("HW device context for 'cuda' initialized successfully");
}

enum AVPixelFormat Decoder::getHWFormat(AVCodecContext* ctx,
                                        const enum AVPixelFormat* pix_fmts)
{
    CELUX_TRACE("getHWFormat called with ctx: {} and pixel formats list");
    for (const enum AVPixelFormat* p = pix_fmts; *p != -1; p++)
    {
        CELUX_DEBUG("Checking pixel format: {}", av_get_pix_fmt_name(*p));
        if (*p == AV_PIX_FMT_CUDA)
        {
            CELUX_INFO("CUDA pixel format found: {}", av_get_pix_fmt_name(*p));
            return *p;
        }
    }
    CELUX_WARN("CUDA pixel format not found in the provided list");
    return AV_PIX_FMT_NONE;
}

} // namespace celux::backends::gpu::cuda
