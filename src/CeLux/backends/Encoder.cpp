#include "backends/cpu/Encoder.hpp"


using namespace celux::error;

namespace celux
{
Encoder::Encoder(std::unique_ptr<celux::conversion::IConverter> converter)
    : converter(std::move(converter)), formatCtx(nullptr), codecCtx(nullptr),
      packet(av_packet_alloc()), stream(nullptr), pts(0)
{
    if (!packet)
    {
        throw CxException("Could not allocate AVPacket");
    }
}

Encoder::~Encoder()
{
    close();
}

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

void Encoder::initialize(const std::string& outputPath, const VideoProperties& props)
{

    properties = props;
    CELUX_DEBUG("Initializing encoder with fps: {}", properties.fps);
    openFile(outputPath, props);
    initHWAccel(); // Virtual function
    const AVCodec* codec = avcodec_find_encoder_by_name(props.codecName.c_str());
    if (!codec)
    {
        throw CxException("Encoder not found: " + props.codecName);
    }
    AVFrame* frame = av_frame_alloc();
    if (!frame)
    {
        throw CxException("Could not allocate AVFrame");
    }
    frame->format = props.pixelFormat;
    frame->width = props.width;
    frame->height = props.height;
    int ret = av_frame_get_buffer(frame, 32);
    if (ret < 0)
    {
        throw CxException("Could not allocate frame buffer");
    }
    this->frame = Frame(frame);
    CELUX_DEBUG("Encoder initialized successfully");
    initCodecContext(codec, props);

    // Create new stream
    stream = avformat_new_stream(formatCtx.get(), codec);
    if (!stream)
    {
        throw CxException("Failed allocating output stream");
    }

    // Copy codec parameters to stream
    ret = avcodec_parameters_from_context(stream->codecpar, codecCtx.get());
    if (ret < 0)
    {
        throw CxException("Failed to copy codec parameters to stream");
    }

    stream->time_base = codecCtx->time_base;

    // Write the stream header
    ret = avformat_write_header(formatCtx.get(), nullptr);
    if (ret < 0)
    {
        throw CxException("Error occurred when writing header to output file");
    }

}

void Encoder::openFile(const std::string& outputPath, const VideoProperties& props)
{
    // Allocate output context
    AVFormatContext* fmt_ctx = nullptr;
    int ret =
        avformat_alloc_output_context2(&fmt_ctx, nullptr, nullptr, outputPath.c_str());
    if (!fmt_ctx)
    {
        throw CxException("Could not allocate output format context");
    }
    formatCtx.reset(fmt_ctx);

    // Open the output file
    if (!(formatCtx->oformat->flags & AVFMT_NOFILE))
    {
        ret = avio_open(&formatCtx->pb, outputPath.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0)
        {
            throw CxException("Could not open output file: " + outputPath);
        }
    }
}

void Encoder::initHWAccel()
{
    // Default implementation does nothing
}

void Encoder::initCodecContext(const AVCodec* codec, const VideoProperties& props)
{
    // Allocate codec context
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx)
    {
        throw CxException("Could not allocate codec context");
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
    
    // Multi-threaded encoding
    codecCtx->thread_count = static_cast<int>(
        std::min(static_cast<unsigned int>(std::thread::hardware_concurrency()), 16u));
    codecCtx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;

    // Allow derived classes to configure additional codec parameters
    configureCodecContext(codec, props);
    // Open the codec
    FF_CHECK(avcodec_open2(codecCtx.get(), codec, nullptr));
}


enum AVPixelFormat Encoder::getHWFormat(AVCodecContext* ctx,
                                        const enum AVPixelFormat* pix_fmts)
{
    // Default implementation returns the first pixel format
    return pix_fmts[0];
}

void Encoder::configureCodecContext(const AVCodec* codec, const VideoProperties& props){
    // do nothing
};

bool Encoder::encodeFrame(void* buffer)
{
    try
    {
        CELUX_DEBUG("Encoding frame...");

        if (!isOpen())
        {
            throw CxException("Encoder is not open");
        }

        // Convert the input buffer to the encoder's frame
        try
        {
            converter->convert(frame, buffer);
            CELUX_DEBUG("Frame converted to encoder's pixel format");
        }
        catch (const std::exception& e)
        {
            throw CxException("Error converting frame: " + std::string(e.what()));
        }

        // Set PTS (Presentation Timestamp)
        frame.get()->pts = pts++;
        CELUX_DEBUG("Frame PTS set to: {}", frame.get()->pts);

        // Rescale PTS to codec's time base
        frame.get()->pts =
            av_rescale_q(frame.get()->pts, {1, static_cast<int>(properties.fps)},
                         codecCtx->time_base);
        CELUX_DEBUG("Scaled Frame PTS: {}", frame.get()->pts);

        // Set packet PTS
        packet->pts = frame.get()->pts;
        CELUX_DEBUG("Packet PTS: {}", packet->pts);

        // Send frame to encoder
        int ret = avcodec_send_frame(codecCtx.get(), frame.get());
        if (ret < 0)
        {
            if (ret == AVERROR(EAGAIN))
            {
                CELUX_DEBUG("EAGAIN encountered. Draining encoder...");
                // Drain the encoder
                while (true)
                {
                    ret = avcodec_receive_packet(codecCtx.get(), packet);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                    {
                        break;
                    }
                    else if (ret < 0)
                    {
                        throw CxException("Error during encoding while draining: " +
                                          celux::errorToString(ret));
                    }

                    // Rescale PTS and DTS
                    av_packet_rescale_ts(packet, codecCtx->time_base,
                                         stream->time_base);
                    packet->stream_index = stream->index;

                    // Write packet
                    ret = av_interleaved_write_frame(formatCtx.get(), packet);
                    if (ret < 0)
                    {
                        av_packet_unref(packet);
                        throw CxException(
                            "Error writing packet during EAGAIN handling: " +
                            celux::errorToString(ret));
                    }

                    av_packet_unref(packet);
                }

                // Retry sending frame after draining
                ret = avcodec_send_frame(codecCtx.get(), frame.get());
                if (ret < 0)
                {
                    throw CxException("Error sending frame after draining: " +
                                      celux::errorToString(ret));
                }
            }
            else
            {
                throw CxException("Error sending frame to encoder: " +
                                  celux::errorToString(ret));
            }
        }

        CELUX_DEBUG("Frame sent to encoder successfully");

        // Receive packets from encoder
        while (true)
        {
            ret = avcodec_receive_packet(codecCtx.get(), packet);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            {
                CELUX_DEBUG("No more packets to receive at this time.");
                break;
            }
            else if (ret < 0)
            {
                throw CxException("Error during encoding: " +
                                  celux::errorToString(ret));
            }

            CELUX_DEBUG("Packet received from encoder");

            // Rescale PTS and DTS to stream time base
            av_packet_rescale_ts(packet, codecCtx->time_base, stream->time_base);
            packet->stream_index = stream->index;

            CELUX_DEBUG("Rescaled packet PTS: {}", packet->pts);

            // Write packet to output file
            ret = av_interleaved_write_frame(formatCtx.get(), packet);
            if (ret < 0)
            {
                av_packet_unref(packet);
                throw CxException("Error writing packet to output file: " +
                                  celux::errorToString(ret));
            }

            av_packet_unref(packet);
        }

        return true;
    }
    catch (const CxException& e)
    {
        CELUX_DEBUG("Error in encodeFrame: {}", e.what());
        return false;
    }
    catch (const std::exception& e)
    {
        CELUX_DEBUG("Error in encodeFrame: {}", e.what());
        return false;
    }
}

bool Encoder::finalize()
{
    try
    {
        CELUX_DEBUG("Finalizing encoder...");

        if (!isOpen())
        {
            CELUX_WARN("Encoder is not open. Nothing to finalize.");
            return false;
        }

        // Send a NULL frame to signal end of stream
        int ret = avcodec_send_frame(codecCtx.get(), nullptr);
        if (ret < 0)
        {
            throw CxException("Error sending flush frame to encoder: " +
                              celux::errorToString(ret));
        }

        // Receive remaining packets
        while (true)
        {
            ret = avcodec_receive_packet(codecCtx.get(), packet);
            if (ret == AVERROR_EOF)
            {
                CELUX_DEBUG("Encoder has been fully flushed.");
                break;
            }
            else if (ret == AVERROR(EAGAIN))
            {
                CELUX_DEBUG("No more packets to receive during finalization.");
                break;
            }
            else if (ret < 0)
            {
                throw CxException("Error during encoding flush: " +
                                  celux::errorToString(ret));
            }

            // Rescale PTS and DTS to stream time base
            av_packet_rescale_ts(packet, codecCtx->time_base, stream->time_base);
            packet->stream_index = stream->index;

            // Write packet
            ret = av_interleaved_write_frame(formatCtx.get(), packet);
            if (ret < 0)
            {
                av_packet_unref(packet);
                throw CxException("Error writing flushed packet to output file: " +
                                  celux::errorToString(ret));
            }

            av_packet_unref(packet);
        }

        // Write the trailer to finalize the file
        ret = av_write_trailer(formatCtx.get());
        if (ret < 0)
        {
            throw CxException("Error writing trailer to output file: " +
                              celux::errorToString(ret));
        }

        CELUX_DEBUG("Encoder finalized successfully");
        return true;
    }
    catch (const CxException& e)
    {
        CELUX_DEBUG("Error in finalize(): {}", e.what());
        return false;
    }
    catch (const std::exception& e)
    {
        CELUX_DEBUG("Error in finalize(): {}", e.what());
        return false;
    }
}

bool Encoder::isOpen() const
{
    return formatCtx && codecCtx;
}

void Encoder::close()
{
    if (converter)
    {
        CELUX_DEBUG("Synchronizing converter in Encoder Destructor");
        converter->synchronize();
        converter.reset();
    }
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
    //stream = nullptr;
}

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

AVCodecContext* Encoder::getCtx()
{
    return codecCtx.get();
}

int64_t Encoder::convertTimestamp(double timestamp) const
{
    AVRational time_base = stream->time_base;
    return static_cast<int64_t>(timestamp * time_base.den / time_base.num);
}

} // namespace celux
