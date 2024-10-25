// backends/cpu/Encoder.cpp

#include "backends/cpu/Encoder.hpp"
#include <Factory.hpp>
using namespace celux::error;

namespace celux
{

Encoder::Encoder(const std::string& outputPath, int width, int height, double fps,
                 celux::EncodingFormats format, const std::string& codecName,
                 std::optional<torch::Stream> stream)
    : outputPath(outputPath), width(width), height(height), fps(fps), format(format),
      codecName(codecName), converter(nullptr), formatCtx(nullptr), codecCtx(nullptr),
      packet(av_packet_alloc()), stream(nullptr), pts(0), encoderStream(stream)
{
    CELUX_DEBUG("Encoder constructor called: Allocating AVPacket");
    if (!packet)
    {
        CELUX_ERROR("Could not allocate AVPacket in Encoder constructor");
        throw CxException("Could not allocate AVPacket");
    }
    CELUX_INFO("AVPacket allocated successfully in Encoder constructor");
}

Encoder::~Encoder()
{
    CELUX_DEBUG("Encoder destructor called");
    close();
    CELUX_INFO("Encoder destructor completed");
}

Encoder::Encoder(Encoder&& other) noexcept
    : formatCtx(std::move(other.formatCtx)), codecCtx(std::move(other.codecCtx)),
      hwDeviceCtx(std::move(other.hwDeviceCtx)),
      hwFramesCtx(std::move(other.hwFramesCtx)), stream(other.stream),
      packet(other.packet), hwAccelType(std::move(other.hwAccelType)), pts(other.pts),
      converter(std::move(other.converter))
{
    CELUX_DEBUG("Encoder move constructor called: Transferring resources");
    other.stream = nullptr;
    other.packet = nullptr;
    CELUX_INFO("Encoder move constructor completed: Resources transferred");
}

Encoder& Encoder::operator=(Encoder&& other) noexcept
{
    CELUX_DEBUG("Encoder move assignment operator called");
    if (this != &other)
    {
        CELUX_DEBUG("Encoder move assignment: Closing existing resources");
        close();

        CELUX_DEBUG("Encoder move assignment: Transferring resources from other");
        formatCtx = std::move(other.formatCtx);
        codecCtx = std::move(other.codecCtx);
        hwDeviceCtx = std::move(other.hwDeviceCtx);
        hwFramesCtx = std::move(other.hwFramesCtx);
        stream = other.stream;
        packet = other.packet;
        hwAccelType = std::move(other.hwAccelType);
        pts = other.pts;
        converter = std::move(other.converter);

        other.stream = nullptr;
        other.packet = nullptr;
        CELUX_INFO("Encoder move assignment completed: Resources transferred");
    }
    else
    {
        CELUX_WARN("Self-assignment detected in Encoder move assignment operator");
    }
    return *this;
}

void Encoder::initialize()
{
    CELUX_INFO("Initializing Encoder with outputPath: {}", outputPath);
    CELUX_INFO("Video Properties - FPS: {}, Width: {}, Height: {}, PixelFormat: {}, "
               "CodecName: {}",
               fps, width, height, celux::encoderFormatToString(format), codecName);

    CELUX_DEBUG("Initializing encoder with fps: {}", fps);
    openFile();
    CELUX_DEBUG("File opened successfully in initialize()");

    initHWAccel(); // Virtual function
    CELUX_DEBUG("Hardware acceleration initialized (if applicable)");

    const AVCodec* codec = avcodec_find_encoder_by_name(codecName.c_str());
    if (!codec)
    {
        CELUX_ERROR("Encoder not found: {}", codecName);
        throw CxException("Encoder not found: " + codecName);
    }
    CELUX_INFO("Found encoder: {}", codec->name);

    AVFrame* frame = av_frame_alloc();
    if (!frame)
    {
        CELUX_ERROR("Could not allocate AVFrame in initialize()");
        throw CxException("Could not allocate AVFrame");
    }
    CELUX_DEBUG("AVFrame allocated successfully in initialize()");

    frame->format = celux::getEncoderPixelFormat(format);
    frame->width = width;
    frame->height = height;
    CELUX_TRACE("Setting AVFrame properties - Format: {}, Width: {}, Height: {}",
                frame->format, frame->width, frame->height);

    int ret = av_frame_get_buffer(frame, 32);
    if (ret < 0)
    {
        CELUX_ERROR("Could not allocate frame buffer: {}", celux::errorToString(ret));
        throw CxException("Could not allocate frame buffer");
    }
    CELUX_DEBUG("Frame buffer allocated with alignment 32");

    this->frame = Frame(frame);
    CELUX_INFO("Frame initialized successfully in initialize()");

    initCodecContext(codec);
    CELUX_DEBUG("Codec context initialized successfully");

    // Create new stream
    CELUX_DEBUG("Creating new stream for the output file");
    stream = avformat_new_stream(formatCtx.get(), codec);
    if (!stream)
    {
        CELUX_ERROR("Failed allocating output stream");
        throw CxException("Failed allocating output stream");
    }
    CELUX_INFO("Output stream created successfully");

    // Copy codec parameters to stream
    ret = avcodec_parameters_from_context(stream->codecpar, codecCtx.get());
    if (ret < 0)
    {
        CELUX_ERROR("Failed to copy codec parameters to stream: {}",
                    celux::errorToString(ret));
        throw CxException("Failed to copy codec parameters to stream");
    }
    CELUX_DEBUG("Codec parameters copied to stream successfully");

    stream->time_base = codecCtx->time_base;
    CELUX_TRACE("Stream time_base set to: {}/{}", stream->time_base.num,
                stream->time_base.den);

    // Write the stream header
    ret = avformat_write_header(formatCtx.get(), nullptr);
    if (ret < 0)
    {
        CELUX_ERROR("Error occurred when writing header to output file: {}",
                    celux::errorToString(ret));
        throw CxException("Error occurred when writing header to output file");
    }

    CELUX_DEBUG("Creating Converter");
    converter = Factory::createConverter(hwDeviceCtx ? torch::kCUDA : torch::kCPU,
                                         celux::getConverterPixelFormat(format),
                                         encoderStream);

    CELUX_INFO("Stream header written successfully");
}

void Encoder::openFile()
{
    CELUX_INFO("Opening output file: {}", outputPath);
    // Allocate output context
    AVFormatContext* fmt_ctx = nullptr;
    int ret =
        avformat_alloc_output_context2(&fmt_ctx, nullptr, nullptr, outputPath.c_str());
    if (!fmt_ctx)
    {
        CELUX_ERROR("Could not allocate output format context for file: {}",
                    outputPath);
        throw CxException("Could not allocate output format context");
    }
    formatCtx.reset(fmt_ctx);
    CELUX_DEBUG("Output format context allocated successfully");

    // Open the output file
    if (!(formatCtx->oformat->flags & AVFMT_NOFILE))
    {
        CELUX_DEBUG("Opening output file with AVIO: {}", outputPath);
        ret = avio_open(&formatCtx->pb, outputPath.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0)
        {
            CELUX_ERROR("Could not open output file: {} with error: {}", outputPath,
                        celux::errorToString(ret));
            throw CxException("Could not open output file: " + outputPath);
        }
        CELUX_INFO("Output file opened successfully with AVIO: {}", outputPath);
    }
    else
    {
        CELUX_DEBUG("Output format does not require AVIO opening for file: {}",
                    outputPath);
    }
}

void Encoder::initHWAccel()
{
    CELUX_DEBUG("Initializing hardware acceleration (CPU backend - no action taken)");
    // Default implementation does nothing for CPU backend
}

void Encoder::initCodecContext(const AVCodec* codec)
{
    CELUX_INFO("Initializing codec context for codec: {}", codec->name);
    // Allocate codec context
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx)
    {
        CELUX_ERROR("Could not allocate codec context for codec: {}", codec->name);
        throw CxException("Could not allocate codec context");
    }
    codecCtx.reset(codec_ctx);
    CELUX_DEBUG("Codec context allocated successfully");

    // Set codec parameters
    codecCtx->width = width;
    codecCtx->height = height;
    codecCtx->time_base = {1, static_cast<int>(fps)};
    codecCtx->framerate = {static_cast<int>(fps), 1};
    codecCtx->gop_size = 12;
    codecCtx->max_b_frames = 0;
    codecCtx->pix_fmt = celux::getEncoderPixelFormat(format);
    CELUX_TRACE("Codec Context Parameters - Width: {}, Height: {}, Time Base: {}/{}, "
                "Framerate: {}/1, GOP Size: {}, Max B-Frames: {}, Pixel Format: {}",
                codecCtx->width, codecCtx->height, codecCtx->time_base.num,
                codecCtx->time_base.den, codecCtx->framerate.num, codecCtx->gop_size,
                codecCtx->max_b_frames, av_get_pix_fmt_name(codecCtx->pix_fmt));

    // Multi-threaded encoding
    codecCtx->thread_count = static_cast<int>(
        std::min(static_cast<unsigned int>(std::thread::hardware_concurrency()), 16u));
    codecCtx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
    CELUX_DEBUG(
        "Configured codec context threading - thread_count: {}, thread_type: {}",
        codecCtx->thread_count, codecCtx->thread_type);

    // Allow derived classes to configure additional codec parameters
    configureCodecContext(codec);
    CELUX_DEBUG(
        "Configured additional codec context parameters via configureCodecContext");

    // Open the codec
    int ret = avcodec_open2(codecCtx.get(), codec, nullptr);
    if (ret < 0)
    {
        CELUX_ERROR("Failed to open codec {}: {}", codec->name,
                    celux::errorToString(ret));
        throw CxException("Failed to open codec: " + std::string(codec->name));
    }
    CELUX_INFO("Codec {} opened successfully", codec->name);
}

enum AVPixelFormat Encoder::getHWFormat(AVCodecContext* ctx,
                                        const enum AVPixelFormat* pix_fmts)
{
    CELUX_TRACE("getHWFormat() called with AVCodecContext: {} and pixel formats list");
    // Default implementation returns the first pixel format
    CELUX_DEBUG("Returning first pixel format: {}", av_get_pix_fmt_name(pix_fmts[0]));
    return pix_fmts[0];
}

void Encoder::configureCodecContext(const AVCodec* codec)
{
    CELUX_DEBUG(
        "configureCodecContext() called: No additional configuration for CPU backend");
    // do nothing for CPU backend
}

bool Encoder::encodeFrame(void* buffer)
{
    CELUX_TRACE("encodeFrame() called");
    CELUX_DEBUG("Encoding frame...");

    if (!isOpen())
    {
        CELUX_ERROR("Encoder is not open");
        throw CxException("Encoder is not open");
    }

    // Convert the input buffer to the encoder's frame
    try
    {
        CELUX_DEBUG("Converting input buffer to encoder's AVFrame");
        converter->convert(frame, buffer);
        CELUX_INFO("Frame converted to encoder's pixel format successfully");
    }
    catch (const std::exception& e)
    {
        CELUX_ERROR("Error converting frame: {}", e.what());
        throw CxException("Error converting frame: " + std::string(e.what()));
    }

    // Set PTS (Presentation Timestamp)
    frame.get()->pts = pts++;
    CELUX_DEBUG("Frame PTS set to: {}", frame.get()->pts);

    // Rescale PTS to codec's time base
    frame.get()->pts =
        av_rescale_q(frame.get()->pts, {1, static_cast<int>(fps)}, codecCtx->time_base);
    CELUX_DEBUG("Scaled Frame PTS: {}", frame.get()->pts);

    // Set packet PTS
    packet->pts = frame.get()->pts;
    CELUX_DEBUG("Packet PTS set to: {}", packet->pts);

    // Send frame to encoder
    int ret = avcodec_send_frame(codecCtx.get(), frame.get());
    if (ret < 0)
    {
        if (ret == AVERROR(EAGAIN))
        {
            CELUX_WARN("EAGAIN encountered. Draining encoder...");
            // Drain the encoder
            while (true)
            {
                ret = avcodec_receive_packet(codecCtx.get(), packet);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                {
                    CELUX_DEBUG("No more packets to receive during draining");
                    break;
                }
                else if (ret < 0)
                {
                    CELUX_ERROR("Error during encoding while draining: {}",
                                celux::errorToString(ret));
                    throw CxException("Error during encoding while draining: " +
                                      celux::errorToString(ret));
                }

                CELUX_DEBUG("Packet received from encoder during draining");

                // Rescale PTS and DTS
                av_packet_rescale_ts(packet, codecCtx->time_base, stream->time_base);
                packet->stream_index = stream->index;
                CELUX_TRACE("Rescaled packet PTS/DTS to stream time base: {}/{}",
                            packet->pts, packet->dts);

                // Write packet
                ret = av_interleaved_write_frame(formatCtx.get(), packet);
                if (ret < 0)
                {
                    av_packet_unref(packet);
                    CELUX_ERROR("Error writing packet during EAGAIN handling: {}",
                                celux::errorToString(ret));
                    throw CxException("Error writing packet during EAGAIN handling: " +
                                      celux::errorToString(ret));
                }

                CELUX_DEBUG(
                    "Packet written to output file successfully during draining");

                av_packet_unref(packet);
            }

            // Retry sending frame after draining
            CELUX_DEBUG("Retrying to send frame to encoder after draining");
            ret = avcodec_send_frame(codecCtx.get(), frame.get());
            if (ret < 0)
            {
                CELUX_ERROR("Error sending frame to encoder after draining: {}",
                            celux::errorToString(ret));
                throw CxException("Error sending frame to encoder after draining: " +
                                  celux::errorToString(ret));
            }
            CELUX_DEBUG("Frame sent to encoder successfully after draining");
        }
        else
        {
            CELUX_ERROR("Error sending frame to encoder: {}",
                        celux::errorToString(ret));
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
            CELUX_DEBUG("No more packets to receive from encoder at this time");
            break;
        }
        else if (ret < 0)
        {
            CELUX_ERROR("Error during encoding: {}", celux::errorToString(ret));
            throw CxException("Error during encoding: " + celux::errorToString(ret));
        }

        CELUX_DEBUG("Packet received from encoder");

        // Rescale PTS and DTS to stream time base
        av_packet_rescale_ts(packet, codecCtx->time_base, stream->time_base);
        packet->stream_index = stream->index;
        CELUX_TRACE("Rescaled packet PTS/DTS to stream time base: {}/{}", packet->pts,
                    packet->dts);

        // Write packet to output file
        ret = av_interleaved_write_frame(formatCtx.get(), packet);
        if (ret < 0)
        {
            av_packet_unref(packet);
            CELUX_ERROR("Error writing packet to output file: {}",
                        celux::errorToString(ret));
            throw CxException("Error writing packet to output file: " +
                              celux::errorToString(ret));
        }

        CELUX_DEBUG("Packet written to output file successfully");

        av_packet_unref(packet);
    }

    CELUX_INFO("encodeFrame() completed successfully");
    return true;
}

bool Encoder::finalize()
{
    CELUX_INFO("Finalizing encoder...");
    try
    {
        if (!isOpen())
        {
            CELUX_WARN("Encoder is not open. Nothing to finalize.");
            return false;
        }

        // Send a NULL frame to signal end of stream
        CELUX_DEBUG("Sending NULL frame to encoder to flush");
        int ret = avcodec_send_frame(codecCtx.get(), nullptr);
        if (ret < 0)
        {
            CELUX_ERROR("Error sending flush frame to encoder: {}",
                        celux::errorToString(ret));
            return false;
        }

        // Receive remaining packets
        while (true)
        {
            ret = avcodec_receive_packet(codecCtx.get(), packet);
            if (ret == AVERROR_EOF)
            {
                CELUX_DEBUG("Encoder has been fully flushed (AVERROR_EOF)");
                break;
            }
            else if (ret == AVERROR(EAGAIN))
            {
                CELUX_DEBUG("No more packets to receive during finalization");
                break;
            }
            else if (ret < 0)
            {
                CELUX_ERROR("Error during encoding flush: {}",
                            celux::errorToString(ret));
                throw CxException("Error during encoding flush: " +
                                  celux::errorToString(ret));
            }

            CELUX_DEBUG("Packet received from encoder during finalization");

            // Rescale PTS and DTS to stream time base
            av_packet_rescale_ts(packet, codecCtx->time_base, stream->time_base);
            packet->stream_index = stream->index;
            CELUX_TRACE("Rescaled packet PTS/DTS to stream time base: {}/{}",
                        packet->pts, packet->dts);

            // Write packet
            ret = av_interleaved_write_frame(formatCtx.get(), packet);
            if (ret < 0)
            {
                av_packet_unref(packet);
                CELUX_ERROR("Error writing flushed packet to output file: {}",
                            celux::errorToString(ret));
                throw CxException("Error writing flushed packet to output file: " +
                                  celux::errorToString(ret));
            }

            CELUX_DEBUG("Flushed packet written to output file successfully");

            av_packet_unref(packet);
        }

        // Write the trailer to finalize the file
        CELUX_DEBUG("Writing trailer to output file");
        ret = av_write_trailer(formatCtx.get());
        if (ret < 0)
        {
            CELUX_ERROR("Error writing trailer to output file: {}",
                        celux::errorToString(ret));
            throw CxException("Error writing trailer to output file: " +
                              celux::errorToString(ret));
        }
        CELUX_INFO("Encoder finalized successfully");
        return true;
    }
    catch (const CxException& e)
    {
        CELUX_ERROR("Exception in finalize(): {}", e.what());
        return false;
    }
    catch (const std::exception& e)
    {
        CELUX_ERROR("Exception in finalize(): {}", e.what());
        return false;
    }
}

bool Encoder::isOpen() const
{
    bool open = formatCtx && codecCtx;
    CELUX_DEBUG("isOpen() called: {}",
                open ? "Encoder is open" : "Encoder is not open");
    return open;
}

void Encoder::close()
{
    CELUX_INFO("Closing Encoder");
    if (converter)
    {
        CELUX_DEBUG("Synchronizing converter in Encoder close");
        converter->synchronize();
        converter.reset();
        CELUX_INFO("Converter synchronized and reset successfully");
    }
    if (finalize())
    {
        CELUX_INFO("Encoder finalized successfully during close()");
    }
    else
    {
        CELUX_WARN("Encoder finalization failed during close()");
    }
    if (packet)
    {
        av_packet_free(&packet);
        CELUX_DEBUG("AVPacket freed successfully in close()");
        packet = nullptr;
    }

    // Reset smart pointers to free resources
    codecCtx.reset();
    formatCtx.reset();
    hwDeviceCtx.reset();
    hwFramesCtx.reset();
    CELUX_INFO("Encoder resources have been released successfully");
}

std::vector<std::string> Encoder::listSupportedEncoders() const
{
    CELUX_TRACE("listSupportedEncoders() called");
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
            CELUX_TRACE("Supported encoder found: {}", codecInfo);
        }
    }

    CELUX_DEBUG("Total supported encoders found: {}", encoders.size());
    return encoders;
}

AVCodecContext* Encoder::getCtx()
{
    CELUX_TRACE("getCtx() called: Returning AVCodecContext pointer");
    return codecCtx.get();
}

int64_t Encoder::convertTimestamp(double timestamp) const
{
    CELUX_TRACE("convertTimestamp() called with timestamp: {}", timestamp);
    AVRational time_base = stream->time_base;
    int64_t ts = static_cast<int64_t>(timestamp * time_base.den / time_base.num);
    CELUX_DEBUG("Converted timestamp: {} to timestamp base: {}", ts, timestamp);
    return ts;
}

} // namespace celux
