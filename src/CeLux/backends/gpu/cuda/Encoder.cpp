// backends/gpu/cuda/Encoder.cpp

#include "backends/gpu/cuda/Encoder.hpp"
using namespace celux::error;

namespace celux::backends::gpu::cuda
{
void Encoder::initHWAccel()
{
    CELUX_INFO("Initializing hardware acceleration (CUDA backend)");
    CELUX_DEBUG("Searching for HW device type: CUDA");

    // Find the hardware device type
    enum AVHWDeviceType type = av_hwdevice_find_type_by_name("cuda");
    if (type == AV_HWDEVICE_TYPE_NONE)
    {
        CELUX_ERROR("Failed to find HW device type: CUDA");
        throw CxException("Failed to find HW device type: CUDA");
    }
    CELUX_DEBUG("Found CUDA hardware device type: {}", av_hwdevice_get_type_name(type));

    // Initialize hardware device context
    AVBufferRef* hw_ctx = nullptr;
    int ret = av_hwdevice_ctx_create(&hw_ctx, type, nullptr, nullptr, 0);
    if (ret < 0)
    {
        CELUX_ERROR("Failed to create HW device context for CUDA: {}",
                    celux::errorToString(ret));
        throw CxException("Failed to create HW device context: " +
                          celux::errorToString(ret));
    }
    CELUX_DEBUG("Created CUDA hardware device context successfully");
    hwDeviceCtx.reset(hw_ctx);
    CELUX_INFO("CUDA hardware device context initialized and set");

    // Initialize hardware frames context
    AVBufferRef* frames_ctx = av_hwframe_ctx_alloc(hwDeviceCtx.get());
    if (!frames_ctx)
    {
        CELUX_ERROR("Failed to create hardware frames context for CUDA");
        throw CxException("Failed to create hardware frames context");
    }
    CELUX_DEBUG("Created CUDA hardware frames context");

    AVHWFramesContext* frames = reinterpret_cast<AVHWFramesContext*>(frames_ctx->data);
    frames->format = AV_PIX_FMT_CUDA;    // Hardware pixel format
    frames->sw_format = getEncoderPixelFormat(format); // Software pixel format (input format)
    frames->width = width;
    frames->height = height;
    frames->initial_pool_size = 20;
    CELUX_DEBUG("Frame Context Created");
    CELUX_DEBUG("Set CUDA hardware frames context - Format: {}, SW_Format: "
                "{}, Width: {}, Height: {}, Initial Pool Size: {}",
                av_get_pix_fmt_name(frames->format),
                av_get_pix_fmt_name(frames->sw_format), frames->width, frames->height,
                frames->initial_pool_size);

    ret = av_hwframe_ctx_init(frames_ctx);
    if (ret < 0)
    {
        CELUX_ERROR("Failed to initialize hardware frames context for CUDA: {}",
                    celux::errorToString(ret));
        throw CxException("Failed to initialize hardware frames context: " +
                          celux::errorToString(ret));
    }
    CELUX_DEBUG("Initialized CUDA hardware frames context successfully");
    hwFramesCtx.reset(frames_ctx);
    CELUX_INFO("CUDA hardware frames context initialized and set");
}

void Encoder::configureCodecContext(const AVCodec* codec)
{
    CELUX_INFO("Configuring Codec Context with Hardware Acceleration (CUDA)");
    CELUX_DEBUG("Referencing HW device context");

    // Set hardware device and frames context in codec context
    codecCtx->hw_device_ctx = av_buffer_ref(hwDeviceCtx.get());
    if (!codecCtx->hw_device_ctx)
    {
        CELUX_ERROR("Failed to reference HW device context");
        throw CxException("Failed to reference HW device context");
    }
    CELUX_DEBUG("Referenced HW device context successfully");

    CELUX_DEBUG("Referencing HW frames context");
    codecCtx->hw_frames_ctx = av_buffer_ref(hwFramesCtx.get());
    if (!codecCtx->hw_frames_ctx)
    {
        CELUX_ERROR("Failed to reference HW frames context");
        throw CxException("Failed to reference HW frames context");
    }
    CELUX_DEBUG("Referenced HW frames context successfully");

    // Set pixel format to CUDA
    codecCtx->pix_fmt = AV_PIX_FMT_CUDA;
    codecCtx->sw_pix_fmt = getEncoderPixelFormat(format);
    CELUX_DEBUG("Set codec pixel format to CUDA (HW) and NV12 (SW)");

    // Set encoder options (these can be adjusted as needed)
    CELUX_DEBUG("Setting encoder options: preset=p7");
    av_opt_set(codecCtx->priv_data, "preset", "p7", 0); // Example: NVENC preset
    codecCtx->gop_size = 12;
    codecCtx->max_b_frames = 0;
    CELUX_DEBUG("Set codec context GOP size to {}, Max B-frames to {}",
                codecCtx->gop_size, codecCtx->max_b_frames);

    // Configure frame for hardware encoding
    CELUX_DEBUG("Configuring frame for hardware encoding");
    frame.get()->hw_frames_ctx = av_buffer_ref(hwFramesCtx.get());
    if (!frame.get()->hw_frames_ctx)
    {
        CELUX_ERROR("Failed to reference HW frames context in frame");
        throw CxException("Failed to reference HW frames context in frame");
    }
   // frame.get()->format = AV_PIX_FMT_CUDA;
    //CELUX_DEBUG("Set frame pixel format to CUDA");

    // Allocate buffer for frame
    CELUX_DEBUG("Allocating buffer for frame with alignment 32");
    int ret = av_hwframe_get_buffer(hwFramesCtx.get(), frame.get(), 0);
    if (ret < 0)
    {
        CELUX_ERROR("Could not allocate CUDA frame buffer: {}",
                    celux::errorToString(ret));
        throw CxException("Could not allocate CUDA frame buffer");
    }
    CELUX_DEBUG("Allocated CUDA frame buffer successfully");
    frame.get()->width = width;
    frame.get()->height = height;
    CELUX_TRACE("Configured frame dimensions - Width: {}, Height: {}",
                frame.get()->width, frame.get()->height);
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
