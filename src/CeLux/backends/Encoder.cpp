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

bool Encoder::encodeFrame(void* buffer)
{
    try
    {
        CELUX_DEBUG("Encoding frame...");
        if (!isOpen())
        {
            throw CxException("Encoder is not open");
        }

        try
        {
            converter->convert(frame, buffer);
            CELUX_DEBUG("Encoder Successfully converted frame to : {}", av_get_pix_fmt_name(properties.pixelFormat));
        }
        catch (const std::exception& e)
        {
            throw CxException("Error converting frame: " + std::string(e.what()));
        }

        // Set and scale PTS
        frame.get()->pts = pts++;
        CELUX_DEBUG("Frame PTS set to: {}", frame.get()->pts);
        frame.get()->pts = av_rescale_q(frame.get()->pts, {1, static_cast<int>(properties.fps)},
                                  codecCtx->time_base);
        CELUX_DEBUG("Scaled Frame PTS: {}", frame.get()->pts);
        packet->pts = frame.get()->pts;
        CELUX_DEBUG("Packet PTS: {}", packet->pts);
        // Log frame's pixel format
        CELUX_DEBUG("Frame pixel format: " + frame.getPixelFormatString());
        // Send frame to encoder
        int ret = avcodec_send_frame(codecCtx.get(), frame.get());

        if (ret < 0)
        {
            if (ret == AVERROR(EAGAIN))
            {
                std::cerr << "AVERROR(EAGAIN) encountered. Draining encoder..."
                          << std::endl;

                // Drain encoder
                while (true)
                {
                    ret = avcodec_receive_packet(codecCtx.get(), packet);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                    {
                        break;
                    }
                    else if (ret < 0)
                    {
                        std::cerr << "Error during draining encoder: "
                                  << celux::errorToString(ret) << std::endl;
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
                        std::cerr << "Error writing packet during EAGAIN handling: "
                                  << celux::errorToString(ret) << std::endl;
                        throw CxException(
                            "Error writing packet during EAGAIN handling: " +
                            celux::errorToString(ret));
                    }

                    av_packet_unref(packet);
                }

                // Retry sending frame
                ret = avcodec_send_frame(codecCtx.get(), frame.get());
                if (ret < 0)
                {
                    std::cerr << "Error sending frame after draining: "
                              << celux::errorToString(ret) << std::endl;
                    throw CxException("Error sending frame after draining: " +
                                      celux::errorToString(ret));
                }
            }
            else
            {
                std::cerr << "Error sending frame to encoder: "
                          << celux::errorToString(ret) << std::endl;
                throw CxException("Error sending frame to encoder: " +
                                  celux::errorToString(ret));
            }
        }

        // Receive packets from encoder
        while (true)
        {
            ret = avcodec_receive_packet(codecCtx.get(), packet);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            {
                break;
            }
            else if (ret < 0)
            {
                std::cerr << "Error during encoding: " << celux::errorToString(ret)
                          << std::endl;
                throw CxException("Error during encoding: " +
                                  celux::errorToString(ret));
            }

            // Rescale PTS and DTS
            av_packet_rescale_ts(packet, codecCtx->time_base, stream->time_base);
            packet->stream_index = stream->index;

            // Write packet
            ret = av_interleaved_write_frame(formatCtx.get(), packet);
            if (ret < 0)
            {
                av_packet_unref(packet);
                std::cerr << "Error writing packet to output file: "
                          << celux::errorToString(ret) << std::endl;
                throw CxException("Error writing packet to output file: " +
                                  celux::errorToString(ret));
            }

            av_packet_unref(packet);
        }

        return true;
    }
    catch (const CxException& e)
    {
		CELUX_DEBUG("Error in Write Frame: {]", e.what());
		return false;
	}
    catch (const std::exception& e)
    {
        CELUX_DEBUG("Error in Write Frame: " + std::string(e.what()));
        return false;
    }
}

void Encoder::configureCodecContext(const AVCodec* codec, const VideoProperties& props){
    // do nothing
};

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
        throw CxException("Error sending flush frame to encoder");
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
            throw CxException("Error during encoding flush");
        }

        // Rescale PTS and DTS to stream time base
        av_packet_rescale_ts(packet, codecCtx->time_base, stream->time_base);
        packet->stream_index = stream->index;

        // Write the packet
        ret = av_interleaved_write_frame(formatCtx.get(), packet);
        if (ret < 0)
        {
            av_packet_unref(packet);
            throw CxException("Error writing flushed packet to output file");
        }

        av_packet_unref(packet);
    }

    // Write the trailer
    ret = av_write_trailer(formatCtx.get());
    if (ret < 0)
    {
        throw CxException("Error writing trailer to output file");
    }

    return true;
}

bool Encoder::isOpen() const
{
    return formatCtx && codecCtx;
}

void Encoder::close()
{
    if (converter)
    {
        converter->synchronize();
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
