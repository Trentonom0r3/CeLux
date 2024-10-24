// Decoder.cpp
#include "Decoder.hpp"
#include <Factory.hpp>
using namespace celux::error;

namespace celux
{

Decoder::Decoder(std::optional<torch::Stream> stream)
    : converter(nullptr), formatCtx(nullptr), codecCtx(nullptr), pkt(av_packet_alloc()),
      videoStreamIndex(-1), decoderStream(std::move(stream))
{
    CELUX_DEBUG("BASE DECODER: Decoder constructed");
}

Decoder::~Decoder()
{
    CELUX_DEBUG("BASE DECODER: Decoder destructor called");
    close();
}

Decoder::Decoder(Decoder&& other) noexcept
    : formatCtx(std::move(other.formatCtx)), codecCtx(std::move(other.codecCtx)),
      pkt(std::move(other.pkt)), videoStreamIndex(other.videoStreamIndex),
      properties(std::move(other.properties)), frame(std::move(other.frame)),
      converter(std::move(other.converter)), hwDeviceCtx(std::move(other.hwDeviceCtx))
{
    CELUX_DEBUG("BASE DECODER: Decoder move constructor called");
    other.videoStreamIndex = -1;
    // Reset other members if necessary
}

Decoder& Decoder::operator=(Decoder&& other) noexcept
{
    CELUX_DEBUG("BASE DECODER: Decoder move assignment operator called");
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
    CELUX_DEBUG("BASE DECODER: Initializing decoder with file: {}", filePath);
    converter = celux::Factory::createConverter(isHwAccel ? torch::kCUDA : torch::kCPU,
                                                celux::ConversionType::NV12ToRGB,
                                                decoderStream);
    CELUX_DEBUG("BASE DECODER: Converter initialized. HW accel is {}",
               isHwAccel ? "enabled" : "disabled");
    openFile(filePath);
    initHWAccel(); // Initialize HW acceleration (overridden in GPU Decoder)
    findVideoStream();

    const AVCodec* codec =
        avcodec_find_decoder(formatCtx->streams[videoStreamIndex]->codecpar->codec_id);
    if (isHwAccel)
    {
        codec = avcodec_find_decoder_by_name(std::string(codec->name).append("_cuvid").c_str());
    }
    if (!codec)
    {
        CELUX_ERROR("Unsupported codec for file: {}", filePath);
        throw CxException("Unsupported codec!");
    }

    initCodecContext(codec);

    properties.codec = codec->name;
    properties.width = codecCtx->width;
    properties.height = codecCtx->height;
    properties.fps = av_q2d(formatCtx->streams[videoStreamIndex]->avg_frame_rate);

    properties.duration =
        (formatCtx->streams[videoStreamIndex]->duration != AV_NOPTS_VALUE)
            ? static_cast<double>(formatCtx->streams[videoStreamIndex]->duration) *
                  av_q2d(formatCtx->streams[videoStreamIndex]->time_base)
            : 0.0;

    properties.pixelFormat =
        isHwAccel ? codecCtx->sw_pix_fmt
                  : codecCtx->pix_fmt;
    properties.hasAudio = (formatCtx->streams[videoStreamIndex]->codecpar->codec_type ==
                           AVMEDIA_TYPE_AUDIO);
    properties.bitDepth = getBitDepth();

    CELUX_DEBUG("BASE DECODER: Video properties populated");
    const AVPixFmtDescriptor* desc =
        av_pix_fmt_desc_get(isHwAccel ? codecCtx->sw_pix_fmt : codecCtx->pix_fmt);
    if (!desc)
    {
        CELUX_WARN("Unknown pixel format, defaulting to NV12ToRGB");
    }

    int bitDepth = desc->comp[0].depth;
    CELUX_INFO("Pixel format: {}, bit depth: {}",
        av_get_pix_fmt_name(isHwAccel ? codecCtx->sw_pix_fmt : codecCtx->pix_fmt),
			   bitDepth);
    // Calculate total frames if possible
    if (formatCtx->streams[videoStreamIndex]->nb_frames > 0)
    {
        properties.totalFrames = formatCtx->streams[videoStreamIndex]->nb_frames;
    }
    else if (properties.fps > 0 && properties.duration > 0)
    {
        properties.totalFrames = static_cast<int>(properties.fps * properties.duration);
    }
    else
    {
        properties.totalFrames = 0; // Unknown
    }

    CELUX_INFO(
        "Video properties: width={}, height={}, fps={}, duration={}, totalFrames={}",
        properties.width, properties.height, properties.fps, properties.duration,
        properties.totalFrames);

    pkt.reset(av_packet_alloc());
    codecCtx->time_base = formatCtx->streams[videoStreamIndex]->time_base;

    frame = Frame(); // Fallback to CPU Frame

    CELUX_DEBUG("BASE DECODER: Decoder initialization completed");

    CELUX_INFO("BASE DECODER: Decoder using codec: {}, and pixel format: {}",
                codec->name, av_get_pix_fmt_name(codecCtx->pix_fmt));
}

void Decoder::openFile(const std::string& filePath)
{
    CELUX_DEBUG("BASE DECODER: Opening file: {}", filePath);
    // Open input file
    AVFormatContext* fmt_ctx = nullptr;
    int ret = avformat_open_input(&fmt_ctx, filePath.c_str(), nullptr, nullptr);
    if (ret < 0)
    {
        CELUX_ERROR("Failed to open input file: {}", filePath);
        FF_CHECK_MSG(ret, std::string("Failure Opening Input:"));
    }
    formatCtx.reset(fmt_ctx);
    CELUX_DEBUG("BASE DECODER: Input file opened successfully");

    // Retrieve stream information
    ret = avformat_find_stream_info(formatCtx.get(), nullptr);
    if (ret < 0)
    {
        CELUX_ERROR("Failed to find stream info for file: {}", filePath);
        FF_CHECK_MSG(ret, std::string("Failure Finding Stream Info:"));
    }
    CELUX_DEBUG("BASE DECODER: Stream information retrieved successfully");
}

void Decoder::initHWAccel()
{
    CELUX_DEBUG("BASE DECODER: Initializing hardware acceleration");
    // Default implementation does nothing
}

void Decoder::findVideoStream()
{
    CELUX_DEBUG("BASE DECODER: Finding best video stream");
    // Find the best video stream
    int ret =
        av_find_best_stream(formatCtx.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (ret < 0)
    {
        CELUX_ERROR("Could not find any video stream in the input file");
        throw CxException("Could not find any video stream in the input file");
    }
    videoStreamIndex = ret;
    CELUX_DEBUG("BASE DECODER: Video stream found at index {}", videoStreamIndex);
}

void Decoder::initCodecContext(const AVCodec* codec)
{
    CELUX_DEBUG("BASE DECODER: Initializing codec context");
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
    CELUX_DEBUG("BASE DECODER: Codec context allocated");

    // Copy codec parameters from input stream to codec context
    int ret = avcodec_parameters_to_context(
        codecCtx.get(), formatCtx->streams[videoStreamIndex]->codecpar);
    if (ret < 0)
    {
        CELUX_ERROR("Failed to copy codec parameters");
        FF_CHECK_MSG(ret, std::string("Failed to copy codec parameters:"));
    }
    CELUX_DEBUG("BASE DECODER: Codec parameters copied to codec context");
    unsigned int threadCount = 16u;
    // Set hardware device context if available
    if (hwDeviceCtx)
    {
        threadCount = 8u;
        codecCtx->hw_device_ctx = av_buffer_ref(hwDeviceCtx.get());
        if (!codecCtx->hw_device_ctx)
        {
            CELUX_ERROR("Failed to reference HW device context");
            throw CxException("Failed to reference HW device context");
        }
        CELUX_DEBUG("BASE DECODER: Hardware device context set");
    }

    codecCtx->thread_count = static_cast<int>(std::min(
        static_cast<unsigned int>(std::thread::hardware_concurrency()), threadCount));
    codecCtx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
    CELUX_DEBUG("BASE DECODER: Codec context threading configured: thread_count={}, thread_type={}",
                codecCtx->thread_count, codecCtx->thread_type);

    // Open codec
    ret = avcodec_open2(codecCtx.get(), codec, nullptr);
    if (ret < 0)
    {
        CELUX_ERROR("Failed to open codec");
        FF_CHECK_MSG(ret, std::string("Failed to open codec:"));
    }
    CELUX_DEBUG("BASE DECODER: Codec opened successfully");
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
            CELUX_TRACE("No more frames to decode");
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
            CELUX_TRACE("End of file reached, flushing decoder");
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
    CELUX_TRACE("Seeking to timestamp: {}", timestamp);
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
    CELUX_TRACE("Seek successful, codec buffers flushed");

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
    CELUX_DEBUG("BASE DECODER: Decoder isOpen check: {}", open);
    return open;
}

void Decoder::close()
{
    CELUX_DEBUG("BASE DECODER: Closing decoder");
    if (codecCtx)
    {
        codecCtx.reset();
        CELUX_DEBUG("BASE DECODER: Codec context reset");
    }
    if (formatCtx)
    {
        formatCtx.reset();
        CELUX_DEBUG("BASE DECODER: Format context reset");
    }
    if (hwDeviceCtx)
    {
        hwDeviceCtx.reset();
        CELUX_DEBUG("BASE DECODER: Hardware device context reset");
    }
    if (converter)
    {
        CELUX_DEBUG("BASE DECODER: Synchronizing converter in Decoder close");
        converter->synchronize();
        converter.reset();
    }
    videoStreamIndex = -1;
    properties = VideoProperties{};
    CELUX_DEBUG("BASE DECODER: Decoder closed");
}

std::vector<std::string> Decoder::listSupportedDecoders() const
{
    CELUX_DEBUG("BASE DECODER: Listing supported decoders");
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
    CELUX_TRACE("Converted timestamp: {}", ts);
    return ts;
}
int Decoder::getBitDepth() const
{
    CELUX_TRACE("Getting bit depth");
    return av_get_bits_per_pixel(av_pix_fmt_desc_get(isHwAccel ? codecCtx->sw_pix_fmt : codecCtx->pix_fmt));
}

} // namespace celux
