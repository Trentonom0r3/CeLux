// Decoder.cpp
#include "Decoder.hpp"
using namespace celux::error;

namespace celux
{

Decoder::Decoder(std::unique_ptr<celux::conversion::IConverter> converter)
    : converter(std::move(converter)), formatCtx(nullptr), codecCtx(nullptr),
      pkt(av_packet_alloc()), videoStreamIndex(-1)
{
    CELUX_DEBUG("Decoder constructed");
    // Constructor does minimal work
}

Decoder::~Decoder()
{
    CELUX_DEBUG("Decoder destructor called");
    close();
}

Decoder::Decoder(Decoder&& other) noexcept
    : formatCtx(std::move(other.formatCtx)), codecCtx(std::move(other.codecCtx)),
      pkt(std::move(other.pkt)), videoStreamIndex(other.videoStreamIndex),
      properties(std::move(other.properties)), frame(std::move(other.frame)),
      converter(std::move(other.converter)), hwDeviceCtx(std::move(other.hwDeviceCtx))
{
    CELUX_DEBUG("Decoder move constructor called");
    other.videoStreamIndex = -1;
    // Reset other members if necessary
}

Decoder& Decoder::operator=(Decoder&& other) noexcept
{
    CELUX_DEBUG("Decoder move assignment operator called");
    if (this != &other)
    {
        close();

        formatCtx = std::move(other.formatCtx);
        codecCtx = std::move(other.codecCtx);
        pkt = std::move(other.pkt);
        videoStreamIndex = other.videoStreamIndex;
        properties = std::move(other.properties);
        frame = std::move(other.frame);
        converter = std::move(other.converter);
        hwDeviceCtx = std::move(other.hwDeviceCtx);

        other.videoStreamIndex = -1;
        // Reset other members if necessary
    }
    return *this;
}

void Decoder::initialize(const std::string& filePath)
{
    CELUX_INFO("Initializing decoder with file: {}", filePath);
    openFile(filePath);
    initHWAccel(); // Virtual function
    findVideoStream();

    const AVCodec* codec =
        avcodec_find_decoder(formatCtx->streams[videoStreamIndex]->codecpar->codec_id);
    if (!codec)
    {
        CELUX_ERROR("Unsupported codec for file: {}", filePath);
        throw CxException("Unsupported codec!");
    }
    CELUX_DEBUG("Decoder using codec: {}", codec->name);
    initCodecContext(codec);

    // Populate video properties
    VideoProperties vp;
    vp.width = codecCtx->width;
    vp.height = codecCtx->height;
    vp.fps = av_q2d(formatCtx->streams[videoStreamIndex]->avg_frame_rate);
    vp.duration =
        (formatCtx->streams[videoStreamIndex]->duration != AV_NOPTS_VALUE)
            ? static_cast<double>(formatCtx->streams[videoStreamIndex]->duration) *
                  av_q2d(formatCtx->streams[videoStreamIndex]->time_base)
            : 0.0;
    vp.pixelFormat = codecCtx->pix_fmt;
    vp.hasAudio = (formatCtx->streams[videoStreamIndex]->codecpar->codec_type ==
                   AVMEDIA_TYPE_AUDIO);
    CELUX_DEBUG("Video properties populated");

    // Calculate total frames if possible
    if (formatCtx->streams[videoStreamIndex]->nb_frames > 0)
    {
        vp.totalFrames = formatCtx->streams[videoStreamIndex]->nb_frames;
    }
    else if (vp.fps > 0 && vp.duration > 0)
    {
        vp.totalFrames = static_cast<int>(vp.fps * vp.duration);
    }
    else
    {
        vp.totalFrames = 0; // Unknown
    }

    CELUX_DEBUG(
        "Video properties: width={}, height={}, fps={}, duration={}, totalFrames={}",
        vp.width, vp.height, vp.fps, vp.duration, vp.totalFrames);

    properties = vp;

    pkt.reset(av_packet_alloc());
    codecCtx->time_base = formatCtx->streams[videoStreamIndex]->time_base;

    frame = av_frame_alloc();

    CELUX_INFO("Decoder initialization completed");
}

void Decoder::openFile(const std::string& filePath)
{
    CELUX_INFO("Opening file: {}", filePath);
    // Open input file
    AVFormatContext* fmt_ctx = nullptr;
    int ret = avformat_open_input(&fmt_ctx, filePath.c_str(), nullptr, nullptr);
    if (ret < 0)
    {
        CELUX_ERROR("Failed to open input file: {}", filePath);
        FF_CHECK_MSG(ret, std::string("Failure Opening Input:"));
    }
    formatCtx.reset(fmt_ctx);
    CELUX_DEBUG("Input file opened successfully");

    // Retrieve stream information
    ret = avformat_find_stream_info(formatCtx.get(), nullptr);
    if (ret < 0)
    {
        CELUX_ERROR("Failed to find stream info for file: {}", filePath);
        FF_CHECK_MSG(ret, std::string("Failure Finding Stream Info:"));
    }
    CELUX_DEBUG("Stream information retrieved successfully");
}

void Decoder::initHWAccel()
{
    CELUX_DEBUG("Initializing hardware acceleration");
    // Default implementation does nothing
}

void Decoder::findVideoStream()
{
    CELUX_DEBUG("Finding best video stream");
    // Find the best video stream
    int ret =
        av_find_best_stream(formatCtx.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (ret < 0)
    {
        CELUX_ERROR("Could not find any video stream in the input file");
        throw CxException("Could not find any video stream in the input file");
    }
    videoStreamIndex = ret;
    CELUX_DEBUG("Video stream found at index {}", videoStreamIndex);
}

void Decoder::initCodecContext(const AVCodec* codec)
{
    CELUX_DEBUG("Initializing codec context");
    if (!codec)
    {
        CELUX_ERROR("Unsupported codec!");
        throw CxException("Unsupported codec!");
    }

    // Allocate codec context
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx)
    {
        CELUX_ERROR("Could not allocate codec context");
        throw CxException("Could not allocate codec context");
    }
    codecCtx.reset(codec_ctx);
    CELUX_DEBUG("Codec context allocated");

    // Copy codec parameters from input stream to codec context
    int ret = avcodec_parameters_to_context(
        codecCtx.get(), formatCtx->streams[videoStreamIndex]->codecpar);
    if (ret < 0)
    {
        CELUX_ERROR("Failed to copy codec parameters");
        FF_CHECK_MSG(ret, std::string("Failed to copy codec parameters:"));
    }
    CELUX_DEBUG("Codec parameters copied to codec context");

    // Set hardware device context if available
    if (hwDeviceCtx)
    {
        codecCtx->hw_device_ctx = av_buffer_ref(hwDeviceCtx.get());
        if (!codecCtx->hw_device_ctx)
        {
            CELUX_ERROR("Failed to reference HW device context");
            throw CxException("Failed to reference HW device context");
        }
        CELUX_DEBUG("Hardware device context set");
        // codecCtx->get_format = getHWFormat; // Assign the member function
    }

    codecCtx->thread_count = static_cast<int>(
        std::min(static_cast<unsigned int>(std::thread::hardware_concurrency()), 16u));
    codecCtx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
    CELUX_DEBUG("Codec context threading configured: thread_count={}, thread_type={}",
                codecCtx->thread_count, codecCtx->thread_type);

    // Open codec
    ret = avcodec_open2(codecCtx.get(), codec, nullptr);
    if (ret < 0)
    {
        CELUX_ERROR("Failed to open codec");
        FF_CHECK_MSG(ret, std::string("Failed to open codec:"));
    }
    CELUX_DEBUG("Codec opened successfully");
}

enum AVPixelFormat Decoder::getHWFormat(AVCodecContext* ctx,
                                        const enum AVPixelFormat* pix_fmts)
{
    CELUX_DEBUG("Getting hardware pixel format");
    // Default implementation returns the first pixel format
    return pix_fmts[0];
}

bool Decoder::decodeNextFrame(void* buffer)
{
    CELUX_TRACE("Decoding next frame");
    int ret;

    if (buffer == nullptr)
    {
        CELUX_ERROR("Buffer is null");
        throw CxException("Buffer is null");
    }

    while (true)
    {
        // Attempt to receive a decoded frame
        ret = avcodec_receive_frame(codecCtx.get(), frame.get());
        if (ret == AVERROR(EAGAIN))
        {
            CELUX_DEBUG("Decoder needs more packets, reading next packet");
            // Need to feed more packets
            // Proceed to read the next packet
        }
        else if (ret == AVERROR_EOF)
        {
            CELUX_INFO("No more frames to decode");
            // No more frames to decode
            return false;
        }
        else if (ret < 0)
        {
            CELUX_ERROR("Error during decoding");
            throw CxException("Error during decoding");
        }
        else
        {
            // Successfully received a frame
            CELUX_DEBUG("Frame decoded successfully");
            converter->convert(frame, buffer);
            CELUX_DEBUG("Frame converted");
            av_frame_unref(frame.get());
            return true;
        }

        // Read the next packet from the video file
        ret = av_read_frame(formatCtx.get(), pkt.get());
        if (ret == AVERROR_EOF)
        {
            CELUX_INFO("End of file reached, flushing decoder");
            // End of file: flush the decoder
            FF_CHECK(avcodec_send_packet(codecCtx.get(), nullptr));
        }
        else if (ret < 0)
        {
            CELUX_ERROR("Error reading frame");
            throw CxException("Error reading frame");
        }
        else
        {
            CELUX_TRACE("Packet read from file, stream_index={}", pkt->stream_index);
            // If the packet belongs to the video stream, send it to the decoder
            if (pkt->stream_index == videoStreamIndex)
            {
                FF_CHECK(avcodec_send_packet(codecCtx.get(), pkt.get()));
                CELUX_DEBUG("Packet sent to decoder");
            }
            else
            {
                CELUX_DEBUG("Packet does not belong to video stream, skipping");
            }
            // Release the packet back to FFmpeg
            av_packet_unref(pkt.get());
        }
    }
}

bool Decoder::seek(double timestamp)
{
    CELUX_INFO("Seeking to timestamp: {}", timestamp);
    if (timestamp < 0 || timestamp > properties.duration)
    {
        CELUX_WARN("Timestamp out of bounds: {}", timestamp);
        return false;
    }

    int64_t ts = convertTimestamp(timestamp);
    int ret =
        av_seek_frame(formatCtx.get(), videoStreamIndex, ts, AVSEEK_FLAG_BACKWARD);
    if (ret < 0)
    {
        CELUX_ERROR("Seek failed to timestamp: {}", timestamp);
        return false;
    }

    // Flush codec buffers
    avcodec_flush_buffers(codecCtx.get());
    CELUX_DEBUG("Seek successful, codec buffers flushed");

    return true;
}

Decoder::VideoProperties Decoder::getVideoProperties() const
{
    CELUX_TRACE("Retrieving video properties");
    return properties;
}

bool Decoder::isOpen() const
{
    bool open = formatCtx != nullptr && codecCtx != nullptr;
    CELUX_DEBUG("Decoder isOpen check: {}", open);
    return open;
}

void Decoder::close()
{
    CELUX_INFO("Closing decoder");
    if (codecCtx)
    {
        codecCtx.reset();
        CELUX_DEBUG("Codec context reset");
    }
    if (formatCtx)
    {
        formatCtx.reset();
        CELUX_DEBUG("Format context reset");
    }
    if (hwDeviceCtx)
    {
        hwDeviceCtx.reset();
        CELUX_DEBUG("Hardware device context reset");
    }
    if (converter)
    {
        CELUX_DEBUG("Synchronizing converter in Decoder close");
        converter->synchronize();
        converter.reset();
    }
    videoStreamIndex = -1;
    properties = VideoProperties{};
    CELUX_INFO("Decoder closed");
}

std::vector<std::string> Decoder::listSupportedDecoders() const
{
    CELUX_DEBUG("Listing supported decoders");
    std::vector<std::string> decoders;
    void* iter = nullptr;
    const AVCodec* codec = nullptr;

    while ((codec = av_codec_iterate(&iter)) != nullptr)
    {
        if (av_codec_is_decoder(codec))
        {
            std::string codecInfo = std::string(codec->name);

            // Append the long name if available
            if (codec->long_name)
            {
                codecInfo += " - " + std::string(codec->long_name);
            }

            decoders.push_back(codecInfo);
            CELUX_TRACE("Supported decoder found: {}", codecInfo);
        }
    }

    return decoders;
}

AVCodecContext* Decoder::getCtx()
{
    CELUX_TRACE("Getting codec context");
    return codecCtx.get();
}

int64_t Decoder::convertTimestamp(double timestamp) const
{
    CELUX_TRACE("Converting timestamp: {}", timestamp);
    AVRational time_base = formatCtx->streams[videoStreamIndex]->time_base;
    int64_t ts = static_cast<int64_t>(timestamp * time_base.den / time_base.num);
    CELUX_DEBUG("Converted timestamp: {}", ts);
    return ts;
}

} // namespace celux
