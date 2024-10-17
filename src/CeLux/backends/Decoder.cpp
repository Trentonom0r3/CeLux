// Decoder.cpp
#include "Decoder.hpp"
using namespace celux::error;
namespace celux
{

Decoder::Decoder(std::unique_ptr<celux::conversion::IConverter> converter)
    : converter(std::move(converter)), formatCtx(nullptr), codecCtx(nullptr),
      pkt(av_packet_alloc()), videoStreamIndex(-1)
{
    // Constructor does minimal work
}

Decoder::~Decoder()
{
    close();
}

Decoder::Decoder(Decoder&& other) noexcept
    : formatCtx(std::move(other.formatCtx)), codecCtx(std::move(other.codecCtx)),
      pkt(std::move(other.pkt)), videoStreamIndex(other.videoStreamIndex),
      properties(std::move(other.properties)), frame(std::move(other.frame)),
      converter(std::move(other.converter)), hwDeviceCtx(std::move(other.hwDeviceCtx))
{
    other.videoStreamIndex = -1;
    // Reset other members if necessary
}

Decoder& Decoder::operator=(Decoder&& other) noexcept
{
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
    openFile(filePath);
    initHWAccel(); // Virtual function
    findVideoStream();

    const AVCodec* codec =
        avcodec_find_decoder(formatCtx->streams[videoStreamIndex]->codecpar->codec_id);
    if (!codec)
    {
        throw CxException("Unsupported codec!");
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
    vp.hasAudio = (formatCtx->streams[videoStreamIndex]->codecpar->codec_type ==
                   AVMEDIA_TYPE_AUDIO);

    // Calculate total frames if possible
    if (vp.fps > 0 && vp.duration > 0)
    {
        vp.totalFrames = vp.duration * vp.fps;
    }
    else
    {
        vp.totalFrames = 0; // Unknown
    }

    properties = vp;
    pkt.reset(av_packet_alloc());
    codecCtx->time_base = formatCtx->streams[videoStreamIndex]->time_base;

    frame = av_frame_alloc();
}

void Decoder::openFile(const std::string& filePath)
{
    // Open input file
    AVFormatContext* fmt_ctx = nullptr;
    FF_CHECK_MSG(avformat_open_input(&fmt_ctx, filePath.c_str(), nullptr, nullptr),
                 std::string("Failure Opening Input:"));
    formatCtx.reset(fmt_ctx);

    // Retrieve stream information
    FF_CHECK_MSG(avformat_find_stream_info(formatCtx.get(), nullptr),
                 std::string("Failure Finding Stream Info:"));
}

void Decoder::initHWAccel()
{
    // Default implementation does nothing
}

void Decoder::findVideoStream()
{
    // Find the best video stream
    int ret =
        av_find_best_stream(formatCtx.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (ret < 0)
    {
        throw CxException("Could not find any video stream in the input file");
    }
    videoStreamIndex = ret;
}

void Decoder::initCodecContext(const AVCodec* codec)
{
    if (!codec)
    {
        throw CxException("Unsupported codec!");
    }

    // Allocate codec context
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx)
    {
        throw CxException("Could not allocate codec context");
    }
    codecCtx.reset(codec_ctx);

    // Copy codec parameters from input stream to codec context
    FF_CHECK_MSG(avcodec_parameters_to_context(
                     codecCtx.get(), formatCtx->streams[videoStreamIndex]->codecpar),
                 std::string("Failed to copy codec parameters:"));

    // Set hardware device context if available
    if (hwDeviceCtx)
    {
        codecCtx->hw_device_ctx = av_buffer_ref(hwDeviceCtx.get());
        if (!codecCtx->hw_device_ctx)
        {
            throw CxException("Failed to reference HW device context");
        }
       // codecCtx->get_format = getHWFormat; // Assign the member function
    }

    // Open codec
    FF_CHECK_MSG(avcodec_open2(codecCtx.get(), codec, nullptr),
                 std::string("Failed to open codec:"));
    codecCtx->thread_count = std::thread::hardware_concurrency();
    codecCtx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
}

enum AVPixelFormat Decoder::getHWFormat(AVCodecContext* ctx,
                                        const enum AVPixelFormat* pix_fmts)
{
    // Default implementation returns the first pixel format
    return pix_fmts[0];
}

bool Decoder::decodeNextFrame(void* buffer)
{
    bool frameProcessed = false;
    if (buffer == nullptr)
    {
        throw CxException("Buffer is null");
    }

    while (!frameProcessed)
    {
        // Attempt to read a packet from the video file
        int ret = av_read_frame(formatCtx.get(), pkt.get());
        if (ret < 0)
        {
            if (ret == AVERROR_EOF)
            {
                // End of file: flush the decoder
                FF_CHECK(avcodec_send_packet(codecCtx.get(), nullptr));
            }
            else
            {
                throw CxException("Error reading frame");
            }
        }
        else
        {
            // If the packet belongs to the video stream, send it to the decoder
            if (pkt->stream_index == videoStreamIndex)
            {
                FF_CHECK(avcodec_send_packet(codecCtx.get(), pkt.get()));
            }
            // Release the packet back to FFmpeg
            av_packet_unref(pkt.get());
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
                return false;
            }
            else if (ret < 0)
            {
                throw CxException("Error during decoding");
            }

                converter->convert(frame, buffer);

            frameProcessed = true;
            break; // Successfully processed one frame
        }

        // Exit the outer loop if a frame was processed
        if (frameProcessed)
        {
            break;
        }
    }
    av_packet_unref(pkt.get());
    av_frame_unref(frame.get());
    return frameProcessed;
}

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

Decoder::VideoProperties Decoder::getVideoProperties() const
{
    return properties;
}

bool Decoder::isOpen() const
{
    return formatCtx != nullptr && codecCtx != nullptr;
}

void Decoder::close()
{
    if (codecCtx)
    {
        codecCtx.reset();
    }
    if (formatCtx)
    {
        formatCtx.reset();
    }
    if (hwDeviceCtx)
    {
        hwDeviceCtx.reset();
    }
    videoStreamIndex = -1;
    properties = VideoProperties{};
}

std::vector<std::string> Decoder::listSupportedDecoders() const
{
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
        }
    }

    return decoders;
}

AVCodecContext* Decoder::getCtx()
{
    return codecCtx.get();
}

int64_t Decoder::convertTimestamp(double timestamp) const
{
    AVRational time_base = formatCtx->streams[videoStreamIndex]->time_base;
    return static_cast<int64_t>(timestamp * time_base.den / time_base.num);
}

} // namespace celux
