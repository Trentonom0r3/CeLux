#include "backends/gpu/cuda/Encoder.hpp"
using namespace celux::error;

namespace celux::backends::gpu::cuda
{
void Encoder::initHWAccel()
{
    CELUX_DEBUG("Initializing HW Acceleration\n");
    // Find the hardware device type
    enum AVHWDeviceType type = av_hwdevice_find_type_by_name("cuda");
    if (type == AV_HWDEVICE_TYPE_NONE)
    {
        throw CxException("Failed to find HW device type: CUDA");
    }
    CELUX_DEBUG("Found CUDA hardware device type");

    // Initialize hardware device context
    AVBufferRef* hw_ctx = nullptr;
    FF_CHECK(av_hwdevice_ctx_create(&hw_ctx, type, nullptr, nullptr, 0));
    CELUX_DEBUG("Created CUDA hardware device context");
    hwDeviceCtx.reset(hw_ctx);

    // Initialize hardware frames context
    AVBufferRef* frames_ctx = av_hwframe_ctx_alloc(hwDeviceCtx.get());
    if (!frames_ctx)
    {
        throw CxException("Failed to create hardware frames context");
    }
    CELUX_DEBUG("Created CUDA hardware frames context");
    AVHWFramesContext* frames = reinterpret_cast<AVHWFramesContext*>(frames_ctx->data);
    frames->format = AV_PIX_FMT_CUDA;    // Hardware pixel format
    frames->sw_format = AV_PIX_FMT_NV12; // Software pixel format (input format)
    frames->width = properties.width;
    frames->height = properties.height;
    frames->initial_pool_size = 20;
    CELUX_DEBUG("Set CUDA hardware frames context properties");
    int ret = av_hwframe_ctx_init(frames_ctx);
    if (ret < 0)
    {
        throw CxException("Failed to initialize hardware frames context: " +
                          celux::errorToString(ret));
    }

    hwFramesCtx.reset(frames_ctx);
}

void Encoder::configureCodecContext(const AVCodec* codec, const VideoProperties& props)
{
    CELUX_DEBUG("Configuring Codec Context with HW Acceleration\n");

    // Set hardware device and frames context in codec context
    codecCtx->hw_device_ctx = av_buffer_ref(hwDeviceCtx.get());
    if (!codecCtx->hw_device_ctx)
    {
        throw CxException("Failed to reference HW device context");
    }

    codecCtx->hw_frames_ctx = av_buffer_ref(hwFramesCtx.get());
    if (!codecCtx->hw_frames_ctx)
    {
        throw CxException("Failed to reference HW frames context");
    }

    // Set pixel format to CUDA
    codecCtx->pix_fmt = AV_PIX_FMT_CUDA;
    codecCtx->sw_pix_fmt = AV_PIX_FMT_NV12;

    // Set encoder options (these can be adjusted as needed)
    av_opt_set(codecCtx->priv_data, "preset", "p7", 0);  // Example: NVENC preset
    codecCtx->gop_size = 12;
    codecCtx->max_b_frames = 0;

    // Configure frame for hardware encoding
    frame.get()->hw_frames_ctx = av_buffer_ref(hwFramesCtx.get());
    frame.get()->format = AV_PIX_FMT_CUDA;
    

    // Allocate buffer for frame
    FF_CHECK(av_hwframe_get_buffer(hwFramesCtx.get(), frame.get(), 0));
    CELUX_DEBUG("Allocated CUDA frame buffer");
    frame.get()->width = props.width;
    frame.get()->height = props.height;
}

enum AVPixelFormat Encoder::getHWFormat(AVCodecContext* ctx,
                                        const enum AVPixelFormat* pix_fmts)
{
    for (const enum AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; p++)
    {
        if (*p == AV_PIX_FMT_CUDA) // Example: CUDA pixel format
        {
            return *p;
        }
        // Add additional hardware pixel formats if needed
    }

    return AV_PIX_FMT_NONE;
}
} // namespace celux::backends::gpu::cuda
