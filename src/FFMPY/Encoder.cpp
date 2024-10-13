// Encoder.cpp

#include "Encoder.hpp"
#include <algorithm>
#include <stdexcept>
#include <thread>

using namespace ffmpy::error;

namespace ffmpy
{
/**
 * @brief Constructor implementation for Encoder.
 */
Encoder::Encoder(const std::string& outputPath, const VideoProperties& props,
                 bool useHardware, const std::string& hwType,
                 std::unique_ptr<ffmpy::conversion::IConverter> converter)
    : properties(props), hwAccelType(hwType), converter(std::move(converter)),
      packet(av_packet_alloc())
{
    if (!packet)
    {
        throw FFException("Could not allocate AVPacket");
    }

    openFile(outputPath, props, useHardware, hwType);
}

/**
 * @brief Destructor ensures resources are cleaned up.
 */
Encoder::~Encoder()
{
    close();
}

/**
 * @brief Move constructor.
 */
Encoder::Encoder(Encoder&& other) noexcept
    : formatCtx(std::move(other.formatCtx)), codecCtx(std::move(other.codecCtx)),
      hwDeviceCtx(std::move(other.hwDeviceCtx)),
      hwFramesCtx(std::move(other.hwFramesCtx)), stream(other.stream),
      packet(other.packet), properties(other.properties),
      hwAccelType(std::move(other.hwAccelType)), pts(other.pts),
      converter(std::move(other.converter))
{
    other.stream = nullptr;
    other.packet = nullptr;
}

/**
 * @brief Move assignment operator.
 */
Encoder& Encoder::operator=(Encoder&& other) noexcept
{
    if (this != &other)
    {
        close();

        formatCtx = std::move(other.formatCtx);
        codecCtx = std::move(other.codecCtx);
        hwDeviceCtx = std::move(other.hwDeviceCtx);
        hwFramesCtx = std::move(other.hwFramesCtx);
        stream = other.stream;
        packet = other.packet;
        properties = other.properties;
        hwAccelType = std::move(other.hwAccelType);
        pts = other.pts;
        converter = std::move(other.converter);

        other.stream = nullptr;
        other.packet = nullptr;
    }
    return *this;
}

/**
 * @brief Open the output file and initialize encoding contexts.
 */
void Encoder::openFile(const std::string& outputPath, const VideoProperties& props,
                       bool useHardware, const std::string& hwType)
{
    // Allocate output context
    AVFormatContext* fmt_ctx = nullptr;
    int ret =
        avformat_alloc_output_context2(&fmt_ctx, nullptr, nullptr, outputPath.c_str());
    if (!fmt_ctx)
    {
        throw FFException("Could not allocate output format context");
    }
    formatCtx.reset(fmt_ctx);

    // Find the encoder
    const AVCodec* codec = avcodec_find_encoder_by_name(props.codecName.c_str());
    if (!codec)
    {
        throw FFException("Encoder not found: " + props.codecName);
    }

    // Initialize hardware acceleration if enabled
    if (useHardware)
    {
        initHWAccel(hwType);
    }

    // Initialize codec context
    initCodecContext(codec, props);

    // Create new stream
    stream = avformat_new_stream(formatCtx.get(), codec);
    if (!stream)
    {
        throw FFException("Failed allocating output stream");
    }

    // Copy codec parameters to stream
    ret = avcodec_parameters_from_context(stream->codecpar, codecCtx.get());
    if (ret < 0)
    {
        throw FFException("Failed to copy codec parameters to stream");
    }

    stream->time_base = codecCtx->time_base;

    // Open the output file
    if (!(formatCtx->oformat->flags & AVFMT_NOFILE))
    {
        ret = avio_open(&formatCtx->pb, outputPath.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0)
        {
            throw FFException("Could not open output file: " + outputPath);
        }
    }

    // Write the stream header
    ret = avformat_write_header(formatCtx.get(), nullptr);
    if (ret < 0)
    {
        throw FFException("Error occurred when writing header to output file");
    }
}

/**
 * @brief Initialize hardware acceleration contexts.
 */
void Encoder::initHWAccel(const std::string& hwType)
{
    // Find the hardware device type
    enum AVHWDeviceType type = av_hwdevice_find_type_by_name(hwType.c_str());
    if (type == AV_HWDEVICE_TYPE_NONE)
    {
        throw FFException("Failed to find HW device type: " + hwType);
    }

    // Initialize hardware device context
    AVBufferRef* hw_ctx = nullptr;
    FF_CHECK(av_hwdevice_ctx_create(&hw_ctx, type, nullptr, nullptr, 0));

    hwDeviceCtx.reset(hw_ctx);

    // Initialize hardware frames context
    AVBufferRef* frames_ctx = av_hwframe_ctx_alloc(hwDeviceCtx.get());
    if (!frames_ctx)
    {
        throw FFException("Failed to create hardware frames context");
    }

    AVHWFramesContext* frames = reinterpret_cast<AVHWFramesContext*>(frames_ctx->data);
    frames->format = properties.pixelFormat;
    frames->sw_format = AV_PIX_FMT_NV12;
    frames->width = properties.width;
    frames->height = properties.height;
    frames->initial_pool_size = 20;

    int ret = av_hwframe_ctx_init(frames_ctx);
    if (ret < 0)
    {
        throw FFException(ret);
    }

    hwFramesCtx.reset(frames_ctx);
}

/**
 * @brief Initialize the codec context for encoding.
 */
void Encoder::initCodecContext(const AVCodec* codec, const VideoProperties& props)
{
    // Allocate codec context
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx)
    {
        throw FFException("Could not allocate codec context");
    }
    codecCtx.reset(codec_ctx);

    // Set codec parameters
    codecCtx->width = props.width;
    codecCtx->height = props.height;
    codecCtx->time_base = {1, static_cast<int>(props.fps)};
    codecCtx->framerate = {static_cast<int>(props.fps), 1};
    codecCtx->gop_size = 12;
    codecCtx->max_b_frames = 0;
    codecCtx->pix_fmt = props.pixelFormat;

    // If hardware acceleration is enabled, set hw_device_ctx and get_format callback
    if (hwDeviceCtx)
    {
        codecCtx->hw_device_ctx = av_buffer_ref(hwDeviceCtx.get());
        if (!codecCtx->hw_device_ctx)
        {
            throw FFException("Failed to reference HW device context");
        }
        codecCtx->get_format = getHWFormat;
    }

    codecCtx->hw_frames_ctx = av_buffer_ref(hwFramesCtx.get());

    // Set encoder options (these can be adjusted as needed)
    av_opt_set(codecCtx->priv_data, "preset", "p5", 0); // Example: NVENC preset
    av_opt_set(codecCtx->priv_data, "crf", "23", 0);    // Example: CRF value

    // Multi-threaded encoding
    codecCtx->thread_count = static_cast<int>(
        std::min(static_cast<unsigned int>(std::thread::hardware_concurrency()), 16u));
    codecCtx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
    // Convert the raw buffer to AVFrame using the converter
    frame.get()->hw_frames_ctx = av_buffer_ref(hwFramesCtx.get());
    frame.get()->format = AV_PIX_FMT_CUDA;
    FF_CHECK(av_hwframe_get_buffer(hwFramesCtx.get(), frame.get(), 0));
    // Open the codec
    FF_CHECK(avcodec_open2(codecCtx.get(), codec, nullptr));
}

/**
 * @brief Callback to select the hardware pixel format.
 */
enum AVPixelFormat Encoder::getHWFormat(AVCodecContext* ctx,
                                        const enum AVPixelFormat* pix_fmts)
{
    for (const enum AVPixelFormat* p = pix_fmts; *p != -1; p++)
    {
        if (*p == AV_PIX_FMT_CUDA) // Example: CUDA pixel format
        {
            return *p;
        }
        // Add additional hardware pixel formats if needed
    }

    return AV_PIX_FMT_NONE;
}

/**
 * @brief Encode a single frame.
 */
bool Encoder::encodeFrame(void* buffer)
{
    if (!isOpen())
    {
        throw FFException("Encoder is not open");
    }

    try
    {
        converter->convert(frame, buffer);
    }
    catch (const std::exception& e)
    {

        throw FFException("Error Converting Framw");
    }

    // Set PTS
    frame.get()->pts = pts++;

    // Send the frame to the encoder
    int ret = avcodec_send_frame(codecCtx.get(), frame.get());

    if (ret < 0)
    {
        throw FFException(ret);
    }

    // Receive and write packets
    while (ret >= 0)
    {
        ret = avcodec_receive_packet(codecCtx.get(), packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
        {
            break;
        }
        else if (ret < 0)
        {
            throw FFException("Error during encoding");
        }

        // Rescale PTS and DTS to stream time base
        av_packet_rescale_ts(packet, codecCtx->time_base, stream->time_base);
        packet->stream_index = stream->index;

        // Write the packet
        ret = av_interleaved_write_frame(formatCtx.get(), packet);
        if (ret < 0)
        {
            av_packet_unref(packet);
            throw FFException("Error writing packet to output file");
        }

        av_packet_unref(packet);
    }

    return true;
}

/**
 * @brief Finalize the encoding process by flushing the encoder.
 */
bool Encoder::finalize()
{
    if (!isOpen())
    {
        return false;
    }

    // Flush the encoder
    int ret = avcodec_send_frame(codecCtx.get(), nullptr);
    if (ret < 0)
    {
        throw FFException("Error sending flush frame to encoder");
    }

    while (ret >= 0)
    {
        ret = avcodec_receive_packet(codecCtx.get(), packet);
        if (ret == AVERROR_EOF)
        {
            break;
        }
        else if (ret == AVERROR(EAGAIN))
        {
            continue;
        }
        else if (ret < 0)
        {
            throw FFException("Error during encoding flush");
        }

        // Rescale PTS and DTS to stream time base
        av_packet_rescale_ts(packet, codecCtx->time_base, stream->time_base);
        packet->stream_index = stream->index;

        // Write the packet
        ret = av_interleaved_write_frame(formatCtx.get(), packet);
        if (ret < 0)
        {
            av_packet_unref(packet);
            throw FFException("Error writing flushed packet to output file");
        }

        av_packet_unref(packet);
    }

    // Write the trailer
    ret = av_write_trailer(formatCtx.get());
    if (ret < 0)
    {
        throw FFException("Error writing trailer to output file");
    }

    return true;
}

/**
 * @brief Check if the Encoder is successfully initialized and open.
 */
bool Encoder::isOpen() const
{
    return formatCtx && codecCtx;
}

/**
 * @brief Close the encoder and clean up resources.
 */
void Encoder::close()
{
    finalize();
    if (packet)
    {
        av_packet_free(&packet);
        packet = nullptr;
    }

    // Reset smart pointers to free resources
    codecCtx.reset();
    formatCtx.reset();
    hwDeviceCtx.reset();
    hwFramesCtx.reset();
    stream = nullptr;
}

/**
 * @brief List all supported encoders.
 */
std::vector<std::string> Encoder::listSupportedEncoders() const
{
    std::vector<std::string> encoders;
    void* iter = nullptr;
    const AVCodec* codec = nullptr;

    while ((codec = av_codec_iterate(&iter)) != nullptr)
    {
        if (av_codec_is_encoder(codec))
        {
            std::string codecInfo = std::string(codec->name);

            if (codec->long_name)
            {
                codecInfo += " - " + std::string(codec->long_name);
            }

            encoders.push_back(codecInfo);
        }
    }

    return encoders;
}
} // namespace ffmpy