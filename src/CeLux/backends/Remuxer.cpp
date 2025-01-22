/**
 * Remuxer.cpp
 * Implementation of the in-memory remux approach to handle non-interleaved AVI if
 * necessary.
 */

#include "backends/cpu/Remuxer.hpp"
// For avformat_close_input() on input contexts
struct InputFormatDeleter
{
    void operator()(AVFormatContext* ctx) const
    {
        if (ctx)
        {
            avformat_close_input(&ctx); // sets ctx to nullptr
        }
    }
};

// For avformat_free_context() on output contexts
struct OutputFormatDeleter
{
    void operator()(AVFormatContext* ctx) const
    {
        if (ctx)
        {
            avformat_free_context(ctx); // frees & sets ctx to nullptr
        }
    }
};
// -----------------------------------------------------------
// Constructor
// -----------------------------------------------------------
Remuxer::Remuxer(const std::string& filePath) : filePath_(filePath)
{
}

// -----------------------------------------------------------
// Destructor
// -----------------------------------------------------------
Remuxer::~Remuxer()
{
    // Close the final format context
    if (formatCtx_)
    {
        avformat_close_input(&formatCtx_);
        formatCtx_ = nullptr;
    }

    // Free the memory buffer, if any
    freeMemoryResources();
}

// -----------------------------------------------------------
// Public method: prepare()
// -----------------------------------------------------------
AVFormatContext* Remuxer::prepare()
{
    if (triedToPrepare_)
    {
        // If we already tried, just return the result we have (could be nullptr or
        // valid).
        return formatCtx_;
    }
    triedToPrepare_ = true;

    // 1) Detect if we must remux
    if (isNonInterleavedAvi(filePath_))
    {
        // 2) Remux in memory
        int ret = remuxInMemory(filePath_);
        if (ret < 0)
        {
            std::cerr << "[Remuxer] Error remuxing file in memory.\n";
            return nullptr;
        }
        usedMemoryRemux_ = true;

        // 3) Open from that memory buffer
        formatCtx_ = openFromMemoryBuffer();
        if (!formatCtx_)
        {
            std::cerr << "[Remuxer] Failed to openFromMemoryBuffer.\n";
            return nullptr;
        }
    }
    else
    {
        // Not non-interleaved => just open normally
        AVFormatContext* ctx = nullptr;
        int ret = avformat_open_input(&ctx, filePath_.c_str(), nullptr, nullptr);
        if (ret < 0)
        {
            std::cerr << "[Remuxer] avformat_open_input error: " << ret << "\n";
            return nullptr;
        }
        // Retrieve stream info
        ret = avformat_find_stream_info(ctx, nullptr);
        if (ret < 0)
        {
            std::cerr << "[Remuxer] avformat_find_stream_info error: " << ret << "\n";
            avformat_close_input(&ctx);
            return nullptr;
        }

        formatCtx_ = ctx;
    }

    return formatCtx_;
}

// -----------------------------------------------------------
// Check if file is non-interleaved
// -----------------------------------------------------------
// Check if the file is a non-interleaved AVI
bool Remuxer::isNonInterleavedAvi(const std::string& filePath)
{
    AVFormatContext* probeCtx = nullptr;
    if (avformat_open_input(&probeCtx, filePath.c_str(), nullptr, nullptr) < 0)
    {
        CELUX_DEBUG("Failed to open file");
        return false;
    }

    // Ensure it's an AVI file
    bool isAvi = probeCtx->iformat && !strcmp(probeCtx->iformat->name, "avi");
    CELUX_DEBUG("Is the file .avi? {}", isAvi);

    if (!isAvi)
    {
        avformat_close_input(&probeCtx);
        return false;
    }

    // Read packets and check for non-interleaving
    bool nonInterleaved = false;
    int prev_stream_index = -1;
    int consecutive_non_video = 0;

    AVPacket pkt;
    av_init_packet(&pkt);

    for (int i = 0; i < 50; i++) // Read first 50 packets
    {
        if (av_read_frame(probeCtx, &pkt) < 0)
            break;

        if (prev_stream_index != pkt.stream_index)
        {
            prev_stream_index = pkt.stream_index;
            consecutive_non_video = 0; // Reset if streams alternate
        }
        else
        {
            consecutive_non_video++; // Count consecutive packets from the same stream
        }

        if (consecutive_non_video > 10) // Arbitrary threshold to detect non-interleaved
        {
            nonInterleaved = true;
            av_packet_unref(&pkt);
            break;
        }

        av_packet_unref(&pkt);
    }

    CELUX_DEBUG("NonInterleaved? {}", nonInterleaved);

    avformat_close_input(&probeCtx);
    return nonInterleaved;
}



int Remuxer::remuxInMemory(const std::string& inPath)
{
    // This will store negative errors or 0 for success
    int ret = 0;

    // (1) Open input in a unique_ptr with a custom deleter
    std::unique_ptr<AVFormatContext, InputFormatDeleter> ifmt_ctx(nullptr);
    {
        AVFormatContext* rawCtx = nullptr;
        ret = avformat_open_input(&rawCtx, inPath.c_str(), nullptr, nullptr);
        if (ret < 0)
        {
            std::cerr << "[Remuxer] Failed to open input file for remux: " << ret
                      << "\n";
            return ret;
        }
        ifmt_ctx.reset(rawCtx); // Now ifmt_ctx will be auto-closed on scope exit
    }

    // (2) Find stream info
    ret = avformat_find_stream_info(ifmt_ctx.get(), nullptr);
    if (ret < 0)
    {
        std::cerr << "[Remuxer] Failed to find stream info: " << ret << "\n";
        return ret; // unique_ptr destructor calls avformat_close_input(&ifmt_ctx)
    }

    // (3) Allocate output context with a unique_ptr to free it on exit
    std::unique_ptr<AVFormatContext, OutputFormatDeleter> ofmt_ctx(nullptr);
    {
        AVFormatContext* rawOutCtx = nullptr;
        // avformat_alloc_output_context2(...) modifies 'rawOutCtx'
        int allocRet =
            avformat_alloc_output_context2(&rawOutCtx, nullptr, "avi", nullptr);
        // If allocRet < 0 OR rawOutCtx == nullptr => error
        if (allocRet < 0 || !rawOutCtx)
        {
            std::cerr << "[Remuxer] Could not allocate output context: " << allocRet
                      << "\n";
            return (allocRet < 0) ? allocRet : AVERROR_UNKNOWN;
        }
        ofmt_ctx.reset(rawOutCtx);
    }

    // (4) Create output streams to match input streams
    for (unsigned int i = 0; i < ifmt_ctx->nb_streams; i++)
    {
        AVStream* in_stream = ifmt_ctx->streams[i];
        AVStream* out_stream = avformat_new_stream(ofmt_ctx.get(), nullptr);
        if (!out_stream)
        {
            std::cerr << "[Remuxer] Failed to create new out_stream.\n";
            return AVERROR_UNKNOWN;
        }
        // Copy codec parameters
        ret = avcodec_parameters_copy(out_stream->codecpar, in_stream->codecpar);
        if (ret < 0)
        {
            std::cerr << "[Remuxer] avcodec_parameters_copy error: " << ret << "\n";
            return ret;
        }
        out_stream->time_base = in_stream->time_base;
    }

    // (5) Open a dynamic memory buffer
    {
        int dynRet = avio_open_dyn_buf(&ofmt_ctx->pb);
        if (dynRet < 0)
        {
            std::cerr << "[Remuxer] avio_open_dyn_buf failed: " << dynRet << "\n";
            return dynRet; // Freed automatically by unique_ptr destructors
        }
    }

    // (6) Write header
    ret = avformat_write_header(ofmt_ctx.get(), nullptr);
    if (ret < 0)
    {
        std::cerr << "[Remuxer] avformat_write_header error: " << ret << "\n";
        return ret;
    }

    // (7) Read packets from input, write to output
    while (true)
    {
        AVPacket pkt;
        ret = av_read_frame(ifmt_ctx.get(), &pkt);
        if (ret < 0)
        {
            // Either EOF or error
            break;
        }

        AVStream* in_stream = ifmt_ctx->streams[pkt.stream_index];
        AVStream* out_stream = ofmt_ctx->streams[pkt.stream_index];

        // Rescale PTS/DTS
        pkt.pts =
            av_rescale_q_rnd(pkt.pts, in_stream->time_base, out_stream->time_base,
                             (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
        pkt.dts =
            av_rescale_q_rnd(pkt.dts, in_stream->time_base, out_stream->time_base,
                             (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
        pkt.duration =
            av_rescale_q(pkt.duration, in_stream->time_base, out_stream->time_base);
        pkt.pos = -1;

        int wret = av_interleaved_write_frame(ofmt_ctx.get(), &pkt);
        av_packet_unref(&pkt);

        if (wret < 0)
        {
            std::cerr << "[Remuxer] Error writing packet: " << wret << "\n";
            ret = wret;
            break;
        }
    }

    // If 'ret' is AVERROR_EOF, we consider it normal
    if (ret == AVERROR_EOF)
    {
        ret = 0;
    }

    // (8) Write trailer
    {
        int trailerRet = av_write_trailer(ofmt_ctx.get());
        if (trailerRet < 0)
        {
            // If we finished everything else OK, we won't override ret=0 with an error
            // here, but we can log it.
            std::cerr << "[Remuxer] av_write_trailer error: " << trailerRet << "\n";
        }
    }

    // (9) Retrieve the dynamic buffer
    if (ret >= 0)
    {
        int bufSize = avio_close_dyn_buf(ofmt_ctx->pb, &bufferData_);
        ofmt_ctx->pb = nullptr;
        if (bufSize < 0)
        {
            std::cerr << "[Remuxer] avio_close_dyn_buf error: " << bufSize << "\n";
            ret = bufSize;
        }
        else
        {
            bufferSize_ = static_cast<size_t>(bufSize);
        }
    }

    // If everything was successful, ret==0.
    // Otherwise ret < 0 => error => free memory
    if (ret < 0)
    {
        freeMemoryResources();
    }

    // unique_ptr destructors now run for ifmt_ctx and ofmt_ctx,
    // safely closing input & freeing the output context.
    return ret;
}

// -----------------------------------------------------------
// openFromMemoryBuffer()
// -----------------------------------------------------------
static int readPacketFromMemory(void* opaque, uint8_t* buf, int buf_size)
{
    // 'opaque' is our pointer to the vector or raw buffer plus read offset
    // We'll store the data & offset in a small struct so it doesn't conflict across
    // calls For simplicity, let's do it with static variables or something minimal. A
    // real implementation would store a struct with { data*, size, offset } in
    // 'opaque'. We'll assume 'opaque' is a pointer to a struct with that info.

    // This example is simplified. We'll do a static offset for demonstration
    // (not thread-safe!)
    static size_t offset = 0;

    auto memVec = reinterpret_cast<std::vector<unsigned char>*>(opaque);
    size_t remaining = memVec->size() - offset;
    if (remaining == 0)
    {
        return AVERROR_EOF;
    }
    int toRead = (int)((remaining < (size_t)buf_size) ? remaining : (size_t)buf_size);
    if (toRead <= 0)
    {
        return AVERROR_EOF;
    }
    memcpy(buf, memVec->data() + offset, toRead);
    offset += toRead;
    return toRead;
}

AVFormatContext* Remuxer::openFromMemoryBuffer()
{
    // We'll copy the data into a std::vector so we can pass it as the "opaque" pointer
    // for the custom read callback.
    if (!bufferData_ || bufferSize_ == 0)
    {
        return nullptr;
    }

    // We'll allocate a new buffer for avio
    const int avioBufSize = 4096;
    unsigned char* avioBuf = (unsigned char*)av_malloc(avioBufSize);
    if (!avioBuf)
    {
        std::cerr << "[Remuxer] Failed to allocate avioBuf\n";
        return nullptr;
    }

    // Create a vector from our raw buffer
    auto memVec =
        new std::vector<unsigned char>(bufferData_, bufferData_ + bufferSize_);

    // Create custom AVIOContext
    AVIOContext* avioCtx = avio_alloc_context(avioBuf, avioBufSize,
                                              0,      // not writing
                                              memVec, // opaque pointer = our vector
                                              readPacketFromMemory,
                                              nullptr, // no write callback
                                              nullptr  // no seek callback
    );
    if (!avioCtx)
    {
        std::cerr << "[Remuxer] Failed to create avio_alloc_context\n";
        av_free(avioBuf);
        delete memVec;
        return nullptr;
    }

    // Create a new format context
    AVFormatContext* memFmtCtx = avformat_alloc_context();
    if (!memFmtCtx)
    {
        std::cerr << "[Remuxer] Failed to avformat_alloc_context for memory\n";
        avio_context_free(&avioCtx);
        delete memVec;
        return nullptr;
    }

    // Link them
    memFmtCtx->pb = avioCtx;
    memFmtCtx->flags |= AVFMT_FLAG_CUSTOM_IO;

    // Now open input from this custom IO
    if (avformat_open_input(&memFmtCtx, nullptr, nullptr, nullptr) < 0)
    {
        std::cerr << "[Remuxer] avformat_open_input() on memory buffer failed\n";
        avformat_free_context(memFmtCtx);
        avio_context_free(&avioCtx);
        delete memVec;
        return nullptr;
    }

    // We can read stream info
    if (avformat_find_stream_info(memFmtCtx, nullptr) < 0)
    {
        std::cerr << "[Remuxer] avformat_find_stream_info() failed in memory\n";
        avformat_close_input(&memFmtCtx);
        avio_context_free(&avioCtx);
        delete memVec;
        return nullptr;
    }

    // Return the pointer. We keep 'memVec' / 'avioCtx' allocated; Freed on close
    // A fully robust approach might stash them in the Remuxer for eventual cleanup.
    return memFmtCtx;
}

// -----------------------------------------------------------
// freeMemoryResources()
// -----------------------------------------------------------
void Remuxer::freeMemoryResources()
{
    if (bufferData_)
    {
        av_freep(&bufferData_);
        bufferData_ = nullptr;
    }
    bufferSize_ = 0;
}
