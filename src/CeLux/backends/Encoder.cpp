#include "Encoder.hpp"

namespace fs = std::filesystem;

namespace celux
{

Encoder::Encoder(const std::string& filename, const EncodingProperties& properties)
    : properties(properties), filename(filename) // Store filename
{
    initialize();
    openOutputFile();
}

Encoder::~Encoder()
{
    close();
}

void Encoder::initialize()
{
    AVFormatContext* fmt_ctx = nullptr;

    // Infer container format from filename extension
    std::string containerFormat = inferContainerFormat(filename);

    // Allocate format context
    avformat_alloc_output_context2(&fmt_ctx, nullptr, containerFormat.c_str(),
                                   filename.c_str());
    formatCtx.reset(fmt_ctx);

    if (!formatCtx)
    {
        throw std::runtime_error("Unsupported file format inferred: " +
                                 containerFormat);
    }

    validateCodecContainerCompatibility();

    initVideoStream();
    if (properties.audioBitRate > 0)
    {
        initAudioStream();
    }
    pkt.reset(av_packet_alloc());
}
void Encoder::openOutputFile()
{
    // Ensure the parent directory exists, if needed
    filename = normalizePath(filename);
    std::filesystem::path filePath(filename);
    auto parent = filePath.parent_path();
    if (!parent.empty())
    {
        std::error_code ec;
        if (!std::filesystem::create_directories(parent, ec) && ec)
        {
            throw std::runtime_error("Failed to create output directory: " +
                                     parent.string() + " (" + ec.message() + ")");
        }
    }

    // If the file doesn't exist, create it
    if (!std::filesystem::exists(filename))
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Failed to create output file: " + filename);
        }
        file.close();
    }

    if (!(formatCtx->oformat->flags & AVFMT_NOFILE))
    {
        if (avio_open(&formatCtx->pb, filename.c_str(), AVIO_FLAG_WRITE) < 0)
        {
            throw std::runtime_error("Could not open output file: " + filename);
        }
    }

    if (avformat_write_header(formatCtx.get(), nullptr) < 0)
    {
        throw std::runtime_error("Error occurred when writing header");
    }
}

void Encoder::validateCodecContainerCompatibility()
{
    const AVCodec* codec = avcodec_find_encoder_by_name(properties.codec.c_str());
    if (!codec)
    {
        std::cerr << "[Encoder] Failed to open video codec: " << properties.codec
                  << "\n";
        PrintSupportedVideoEncoders(); // <---- print them now!
        throw std::runtime_error("Invalid codec specified: " + properties.codec);
    }

    if (!avformat_query_codec(formatCtx->oformat, codec->id, 0))
    {
        throw std::runtime_error("The codec " + properties.codec +
                                 " is not supported by the inferred container format.");
    }
}


void Encoder::initVideoStream()
{
    videoStream = avformat_new_stream(formatCtx.get(), nullptr);
    if (!videoStream)
    {
        throw std::runtime_error("Failed to create video stream");
    }

    const AVCodec* codec = avcodec_find_encoder_by_name(properties.codec.c_str());
    videoCodecCtx.reset(avcodec_alloc_context3(codec));
    if (!videoCodecCtx)
    {
        throw std::runtime_error("Failed to allocate video codec context");
    }

    videoCodecCtx->bit_rate = properties.bitRate;
    videoCodecCtx->width = properties.width;
    videoCodecCtx->height = properties.height;
    videoCodecCtx->time_base = {1, properties.fps};
    videoCodecCtx->framerate = {properties.fps, 1};
    videoCodecCtx->gop_size = properties.gopSize;
    videoCodecCtx->max_b_frames = properties.maxBFrames;
    videoCodecCtx->pix_fmt = properties.pixelFormat;

    if (avcodec_open2(videoCodecCtx.get(), codec, nullptr) < 0)
    {
        throw std::runtime_error("Failed to open video codec: " + properties.codec);
    }

    avcodec_parameters_from_context(videoStream->codecpar, videoCodecCtx.get());
}

void Encoder::initAudioStream()
{
    audioStream = avformat_new_stream(formatCtx.get(), nullptr);
    if (!audioStream)
        throw std::runtime_error("Failed to create audio stream");

    const AVCodec* codec = avcodec_find_encoder_by_name(properties.audioCodec.c_str());
    audioCodecCtx.reset(avcodec_alloc_context3(codec));
    if (!audioCodecCtx)
        throw std::runtime_error("Failed to allocate audio codec context");

    // Basic parameters
    audioCodecCtx->bit_rate = properties.audioBitRate;
    audioCodecCtx->sample_rate = properties.audioSampleRate;
    audioCodecCtx->ch_layout.nb_channels = properties.audioChannels;
    av_channel_layout_default(&audioCodecCtx->ch_layout, properties.audioChannels);
    audioCodecCtx->time_base = {1, properties.audioSampleRate};

    // Force AAC’s planar‑float format
    audioCodecCtx->sample_fmt = AV_SAMPLE_FMT_FLTP;

    if (avcodec_open2(audioCodecCtx.get(), codec, nullptr) < 0)
        CELUX_WARN("Failed to open audio codec.");

    avcodec_parameters_from_context(audioStream->codecpar, audioCodecCtx.get());
}

bool Encoder::encodeFrame(const Frame& frame)
{
    if (!videoCodecCtx)
        return false;

    // 1) Optionally check if PTS is unset (or negative).
    static int64_t frameCounter = 0;
    AVFrame* avf = frame.get();
    if (avf->pts == AV_NOPTS_VALUE || avf->pts < 0)
    {
        // Assign a strictly increasing PTS.
        avf->pts = frameCounter++;
    }

    // 2) Send the frame to the encoder
    if (int err = avcodec_send_frame(videoCodecCtx.get(), avf); err < 0)
    {
        // Handle error...
        return false;
    }

    // 3) Drain packets from the encoder
    while (avcodec_receive_packet(videoCodecCtx.get(), pkt.get()) == 0)
    {
        writePacket(); // calls av_interleaved_write_frame()
    }
    return true;
}


// In Encoder::encodeAudioFrame
bool Encoder::encodeAudioFrame(const Frame& frame)
{
    if (!audioCodecCtx)
        return false;

    AVFrame* af = frame.get();

    // 1) assign a proper PTS (in units of audioCodecCtx->time_base = 1/sample_rate)
    af->pts = nextAudioPts;
    nextAudioPts += af->nb_samples;

    // 2) send to encoder
    if (avcodec_send_frame(audioCodecCtx.get(), af) < 0)
        return false;

    // 3) pull out packets and mux them immediately
    while (avcodec_receive_packet(audioCodecCtx.get(), pkt.get()) == 0)
    {
        // mark as audio stream
        pkt->stream_index = audioStream->index;

        // rescale audio‐codec time_base -> audio‐stream time_base
        av_packet_rescale_ts(pkt.get(), audioCodecCtx->time_base,
                             audioStream->time_base);

        av_interleaved_write_frame(formatCtx.get(), pkt.get());
    }

    return true;
}

// In writePacket(), simplify to only handle video (we now handle audio above)
void Encoder::writePacket()
{
    // ONLY video packets should ever reach this helper;
    // they already have pkt->stream_index == videoStream->index
    av_packet_rescale_ts(pkt.get(), videoCodecCtx->time_base, videoStream->time_base);
    av_interleaved_write_frame(formatCtx.get(), pkt.get());
}



void Encoder::close()
{
    if (!formatCtx)
        return;

    // Flush video
    if (videoCodecCtx)
    {
        avcodec_send_frame(videoCodecCtx.get(), nullptr);
        while (avcodec_receive_packet(videoCodecCtx.get(), pkt.get()) == 0)
        {
            pkt->stream_index = videoStream->index;
            av_packet_rescale_ts(pkt.get(), videoCodecCtx->time_base,
                                 videoStream->time_base);
            av_interleaved_write_frame(formatCtx.get(), pkt.get());
        }
    }

    // Flush audio
    if (audioCodecCtx)
    {
        avcodec_send_frame(audioCodecCtx.get(), nullptr);
        while (avcodec_receive_packet(audioCodecCtx.get(), pkt.get()) == 0)
        {
            pkt->stream_index = audioStream->index;
            av_packet_rescale_ts(pkt.get(), audioCodecCtx->time_base,
                                 audioStream->time_base);
            av_interleaved_write_frame(formatCtx.get(), pkt.get());
        }
    }

    // Trailer finalizes moov box in .mp4
    av_write_trailer(formatCtx.get());

    // If not NOFILE, close I/O
    if (!(formatCtx->oformat->flags & AVFMT_NOFILE) && formatCtx->pb)
        avio_closep(&formatCtx->pb);

    videoCodecCtx.reset();
    audioCodecCtx.reset();
    formatCtx.reset();
}



/**
 * Infers the container format based on the file extension.
 */
std::string Encoder::inferContainerFormat(const std::string& filename) const
{
    std::string extension = fs::path(filename).extension().string();
    if (extension == ".mp4")
        return "mp4";
    if (extension == ".mkv")
        return "matroska";
    if (extension == ".mov")
        return "mov";
    if (extension == ".webm")
        return "webm";
    if (extension == ".avi")
        return "avi";

    return "mp4"; // Default to MP4 if unknown
}

} // namespace celux
