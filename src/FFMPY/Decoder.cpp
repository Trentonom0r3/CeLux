// Decoder.cpp

#include "Decoder.hpp"


using namespace ffmpy::error;
namespace ffmpy
{
/**
 * @brief Constructor that initializes the Decoder with a video file.
 * @param filePath Path to the input video file.
 * @param useHardware Whether to use hardware acceleration.
 * @param hwType Type of hardware acceleration (e.g., "cuda", "vaapi").
 */
// Constructor implementation
ffmpy::Decoder::Decoder(
    const std::string& filePath, bool useHardware, const std::string& hwType,
    std::unique_ptr<ffmpy::conversion::IConverter> converter)
    : converter(std::move(converter)),
      // Initialize other members as needed
      formatCtx(nullptr), codecCtx(nullptr), hwDeviceCtx(nullptr),
      pkt(av_packet_alloc()), videoStreamIndex(-1), hwAccelType(hwType)
{
    // Initialization logic
    openFile(filePath, useHardware, hwType);
    findVideoStream();

    // Initialize the codec context
    const AVCodec* codec =
        avcodec_find_decoder(formatCtx->streams[videoStreamIndex]->codecpar->codec_id);
    if (!codec)
    {
        throw FFException("Unsupported codec!");
    }

    initCodecContext(codec);

    // Populate video properties
    VideoProperties vp;
    vp.width = codecCtx->width;
    vp.height = codecCtx->height;
    vp.fps = av_q2d(formatCtx->streams[videoStreamIndex]->avg_frame_rate);
    vp.duration = (formatCtx->duration != AV_NOPTS_VALUE)
                      ? static_cast<double>(formatCtx->duration) / AV_TIME_BASE
                      : 0.0;
    vp.pixelFormat = codecCtx->pix_fmt;
    vp.audio = (formatCtx->streams[videoStreamIndex]->codecpar->codec_type ==
                AVMEDIA_TYPE_AUDIO);

    // Calculate total frames if possible
    if (vp.fps > 0 && vp.duration > 0)
    {
        vp.totalFrames = static_cast<int>(vp.duration * vp.fps);
    }
    else
    {
        vp.totalFrames = 0; // Unknown
    }

    properties = vp;
    pkt = av_packet_alloc();
    // **Critical Step:** Set the time_base to match the stream's time_base
    codecCtx->time_base = formatCtx->streams[videoStreamIndex]->time_base;

    frame = av_frame_alloc();
}

std::vector<std::string> Decoder::listSupportedDecoders() const
{
    std::vector<std::string> Decoders;
    void* iter = nullptr; // Iterator for av_codec_iterate
    const AVCodec* codec = nullptr;

    // Iterate through all available codecs
    while ((codec = av_codec_iterate(&iter)) != nullptr)
    {
        // Check if the codec is a Decoder
        if (av_codec_is_decoder(codec))
        {
            std::string codecInfo = std::string(codec->name);

            // Append the long name if available
            if (codec->long_name)
            {
                codecInfo += " - " + std::string(codec->long_name);
            }

            Decoders.push_back(codecInfo);
        }
    }

    return Decoders;
}
/**
 * @brief Destructor that cleans up resources.
 */
Decoder::~Decoder()
{
    close();
}

/**
 * @brief Move constructor that transfers ownership.
 */
Decoder::Decoder(Decoder&& other) noexcept
    : formatCtx(std::move(other.formatCtx)), codecCtx(std::move(other.codecCtx)),
      hwDeviceCtx(std::move(other.hwDeviceCtx)),
      videoStreamIndex(other.videoStreamIndex), properties(other.properties),
      hwAccelType(std::move(other.hwAccelType))
{
    other.videoStreamIndex = -1;
    other.properties = VideoProperties{};
}

/**
 * @brief Move assignment operator that transfers ownership.
 */
Decoder& Decoder::operator=(Decoder&& other) noexcept
{
    if (this != &other)
    {
        close();

        formatCtx = std::move(other.formatCtx);
        codecCtx = std::move(other.codecCtx);
        hwDeviceCtx = std::move(other.hwDeviceCtx);
        videoStreamIndex = other.videoStreamIndex;
        properties = other.properties;
        hwAccelType = std::move(other.hwAccelType);

        other.videoStreamIndex = -1;
        other.properties = VideoProperties{};
    }
    return *this;
}

/**
 * @brief Open the video file and initialize decoding contexts.
 * @param filePath Path to the input video file.
 * @param useHardware Whether to use hardware acceleration.
 * @param hwType Type of hardware acceleration.
 */
void Decoder::openFile(const std::string& filePath, bool useHardware,
                       const std::string& hwType)
{
    // Open input file
    AVFormatContext* fmt_ctx = nullptr;
    FF_CHECK_MSG(avformat_open_input(&fmt_ctx, filePath.c_str(), nullptr, nullptr),
                 std::string("Failure Opening Input:"));
    formatCtx.reset(fmt_ctx);

    // Retrieve stream information
    FF_CHECK_MSG(avformat_find_stream_info(formatCtx.get(), nullptr),
                 std::string("Failure Finding Stream Info:"));

    // Initialize hardware acceleration if enabled
    if (useHardware)
    {
        initHWAccel(hwType);
    }
}

/**
 * @brief Initialize hardware acceleration contexts.
 * @param hwType Type of hardware acceleration.
 */
void Decoder::initHWAccel(const std::string& hwType)
{
    // Find the hardware device type
    enum AVHWDeviceType type = av_hwdevice_find_type_by_name(hwType.c_str());
    if (type == AV_HWDEVICE_TYPE_NONE)
    {
        throw FFException("Failed to find HW device type: " + hwType);
    }

    // Initialize hardware device context
    AVBufferRef* hw_ctx = nullptr;
    FF_CHECK_MSG(av_hwdevice_ctx_create(&hw_ctx, type, nullptr, nullptr, 0),
                 std::string("Failed to create HW device context:"));
    hwDeviceCtx.reset(hw_ctx);
}

/**
 * @brief Find the best video stream in the format context.
 * @throws FFException if no video stream is found.
 */
void Decoder::findVideoStream()
{
    // Find the best video stream
    int ret =
        av_find_best_stream(formatCtx.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (ret < 0)
    {
        throw FFException("Could not find any video stream in the input file");
    }
    videoStreamIndex = ret;
}

/**
 * @brief Initialize the codec context for decoding.
 * @param codec The codec to be used for decoding.
 */
void Decoder::initCodecContext(const AVCodec* codec)
{
    if (!codec)
    {
        throw FFException("Unsupported codec!");
    }

    // Allocate codec context
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx)
    {
        throw FFException("Could not allocate codec context");
    }
    codecCtx.reset(codec_ctx);

    // Copy codec parameters from input stream to codec context
    FF_CHECK_MSG(avcodec_parameters_to_context(
                     codecCtx.get(), formatCtx->streams[videoStreamIndex]->codecpar),
                 std::string("Failed to copy codec parameters:"));

    // Set hardware device context if using hardware acceleration
    if (hwDeviceCtx)
    {
        codecCtx->hw_device_ctx = av_buffer_ref(hwDeviceCtx.get());
        if (!codecCtx->hw_device_ctx)
        {
            throw FFException("Failed to reference HW device context");
        }
        codecCtx->get_format = getHWFormat;
    }

    // Open codec
    FF_CHECK_MSG(avcodec_open2(codecCtx.get(), codec, nullptr),
                 std::string("Failed to open codec:"));
    codecCtx.get()->thread_count =
        std::thread::hardware_concurrency(); // Utilize all available CPU cores
    codecCtx.get()->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
}

/**
 * @brief Decode the next frame from the video stream.
 * @param frame Reference to a Frame object to store the decoded frame.
 * @return True if a frame was successfully decoded, False if end of stream.
 * @throws FFException on decoding errors.
 */
bool Decoder::decodeNextFrame(void* buffer)
{
    bool frameProcessed = false;

    while (!frameProcessed)
    {
        // Attempt to read a packet from the video file
        int ret = av_read_frame(formatCtx.get(), pkt);
        if (ret < 0)
        {
            if (ret == AVERROR_EOF)
            {
                // End of file: flush the Decoder
                avcodec_send_packet(codecCtx.get(), nullptr);
            }
            else
            {
                throw ffmpy::error::FFException(ret);
                break; // Exit the loop on error
            }
        }
        else
        {
            // If the packet belongs to the video stream, send it to the Decoder
            if (pkt->stream_index == videoStreamIndex)
            {
                FF_CHECK(avcodec_send_packet(codecCtx.get(), pkt));
            }
            // Release the packet back to FFmpeg
            av_packet_unref(pkt);
            av_frame_unref(frame.get());
        }

        // Attempt to receive a decoded frame
        while (true)
        {
            ret = avcodec_receive_frame(codecCtx.get(), frame.get());
            if (ret == AVERROR(EAGAIN))
            {
                // Decoder needs more data; proceed to read the next packet
                break;
            }
            else if (ret == AVERROR_EOF)
            {
                // No more frames to decode
                frameProcessed = false;
                break;
            }
            else if (ret < 0)
            {
                throw ffmpy::error::FFException(ret);
                break; // Exit the inner loop on error
            }
            converter->convert(frame, buffer);
            frameProcessed = true;
            break; // Successfully processed one frame
        }

        // Exit the outer loop if a frame was processed or end of stream is reached
        if (frameProcessed || ret == AVERROR_EOF)
        {
            break;
        }
    }
    av_packet_unref(pkt);
    av_frame_unref(frame.get());
    return frameProcessed;
}

/**
 * @brief Seek to a specific timestamp in the video.
 * @param timestamp Timestamp in seconds to seek to.
 * @return True if seeking was successful, False otherwise.
 */
bool Decoder::seek(double timestamp)
{
    if (timestamp < 0 || timestamp > properties.duration)
    {
        return false;
    }

    int64_t ts = convertTimestamp(timestamp);
    int ret =
        av_seek_frame(formatCtx.get(), videoStreamIndex, ts, AVSEEK_FLAG_BACKWARD);
    if (ret < 0)
    {

        return false;
    }

    // Flush codec buffers
    avcodec_flush_buffers(codecCtx.get());

    return true;
}

/**
 * @brief Get video properties like width, height, FPS, etc.
 * @return A struct containing video properties.
 */
Decoder::VideoProperties Decoder::getVideoProperties() const
{
    return properties;
}

/**
 * @brief Check if the Decoder is successfully initialized and open.
 * @return True if open, False otherwise.
 */
bool Decoder::isOpen() const
{
    return formatCtx != nullptr && codecCtx != nullptr;
}

/**
 * @brief Close the video file and clean up resources.
 */
void Decoder::close()
{
    codecCtx.reset();
    formatCtx.reset();
    hwDeviceCtx.reset();
    videoStreamIndex = -1;
    properties = VideoProperties{};
    //AVPacket* pkt
    av_packet_free(&pkt);
}

/**
 * @brief Convert timestamp in seconds to AV_TIME_BASE units.
 * @param timestamp Timestamp in seconds.
 * @return Corresponding timestamp in AV_TIME_BASE units.
 */
int64_t Decoder::convertTimestamp(double timestamp) const
{
    return static_cast<int64_t>(timestamp * AV_TIME_BASE);
}

AVBufferRef* Decoder::getHWDeviceCtx() const
{
    return hwDeviceCtx.get();
}


/**
 * @brief Callback to select the hardware pixel format.
 * @param ctx Codec context.
 * @param pix_fmts List of supported pixel formats.
 * @return Selected pixel format.
 */
enum AVPixelFormat Decoder::getHWFormat(AVCodecContext* ctx,
                                        const enum AVPixelFormat* pix_fmts)
{
    // Iterate through the supported pixel formats
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
} // namespace ffmpy
