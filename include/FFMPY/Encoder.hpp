// Encoder.hpp
#pragma once

#include "FFException.hpp"
#include <Frame.hpp> 
#include <Conversion.hpp>

namespace ffmpy
{
class Encoder
{
  public:
    struct VideoProperties
    {
        int width;
        int height;
        double fps;
        AVPixelFormat pixelFormat;
        std::string codecName;
    };

    /**
     * @brief Constructor to initialize the Encoder with output settings.
     * @param outputPath Path to the output video file.
     * @param props Video properties such as width, height, fps, etc.
     * @param useHardware Whether to use hardware acceleration.
     * @param hwType Type of hardware acceleration (e.g., "cuda", "vaapi").
     * @param converter Unique pointer to a converter for frame format conversions.
     */
    Encoder(const std::string& outputPath, const VideoProperties& props,
            bool useHardware = true, const std::string& hwType = "cuda",
            std::unique_ptr<ffmpy::conversion::IConverter> converter = nullptr);

    virtual ~Encoder();

    // Deleted copy constructor and assignment operator
    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;

    // Move constructor and assignment operator
    Encoder(Encoder&&) noexcept;
    Encoder& operator=(Encoder&&) noexcept;

    /**
     * @brief Encode a single frame.
     * @param buffer Pointer to the raw frame data.
     * @return True if the frame was successfully encoded, False otherwise.
     * @throws FFException on encoding errors.
     */
    bool encodeFrame(void* buffer);

    /**
     * @brief Finalize the encoding process by flushing the encoder.
     * @return True if finalization was successful, False otherwise.
     */
    bool finalize();

    /**
     * @brief Check if the Encoder is successfully initialized and open.
     * @return True if open, False otherwise.
     */
    bool isOpen() const;

    /**
     * @brief Close the encoder and clean up resources.
     */
    void close();

    /**
     * @brief List all supported encoders.
     * @return A vector of supported encoder names and descriptions.
     */
    std::vector<std::string> listSupportedEncoders() const;

    /**
     * @brief Get the underlying codec context.
     * @return Pointer to AVCodecContext.
     */
    AVCodecContext* getCtx()
    {
        return codecCtx.get();
    }

  private:
    /**
     * @brief Open the output file and initialize encoding contexts.
     * @param outputPath Path to the output video file.
     * @param props Video properties.
     * @param useHardware Whether to use hardware acceleration.
     * @param hwType Type of hardware acceleration.
     */
    void openFile(const std::string& outputPath, const VideoProperties& props,
                  bool useHardware, const std::string& hwType);

    /**
     * @brief Initialize hardware acceleration contexts.
     * @param hwType Type of hardware acceleration.
     */
    void initHWAccel(const std::string& hwType);

    /**
     * @brief Initialize the codec context for encoding.
     * @param codec The codec to be used for encoding.
     * @param props Video properties.
     */
    void initCodecContext(const AVCodec* codec, const VideoProperties& props);

    /**
     * @brief Convert timestamp in seconds to AV_TIME_BASE units.
     * @param timestamp Timestamp in seconds.
     * @return Corresponding timestamp in AV_TIME_BASE units.
     */
    int64_t convertTimestamp(double timestamp) const;

    /**
     * @brief Callback to select the hardware pixel format.
     * @param ctx Codec context.
     * @param pix_fmts List of supported pixel formats.
     * @return Selected pixel format.
     */
    static enum AVPixelFormat getHWFormat(AVCodecContext* ctx,
                                          const enum AVPixelFormat* pix_fmts);

    // Custom deleters for smart pointers
    struct AVFormatContextDeleter
    {
        void operator()(AVFormatContext* ctx) const
        {
            if (!(ctx->oformat->flags & AVFMT_NOFILE))
                avio_closep(&ctx->pb);
            avformat_free_context(ctx);
        }
    };

    struct AVCodecContextDeleter
    {
        void operator()(AVCodecContext* ctx) const
        {
            avcodec_free_context(&ctx);
        }
    };

    struct AVBufferRefDeleter
    {
        void operator()(AVBufferRef* ref) const
        {
            av_buffer_unref(&ref);
        }
    };

    using AVFormatContextPtr = std::unique_ptr<AVFormatContext, AVFormatContextDeleter>;
    using AVCodecContextPtr = std::unique_ptr<AVCodecContext, AVCodecContextDeleter>;
    using AVBufferRefPtr = std::unique_ptr<AVBufferRef, AVBufferRefDeleter>;

    // FFmpeg contexts
    AVFormatContextPtr formatCtx;
    AVCodecContextPtr codecCtx;
    AVBufferRefPtr hwDeviceCtx;
    AVBufferRefPtr hwFramesCtx;

    // Encoding stream
    AVStream* stream = nullptr;

    // Packet for encoding
    AVPacket* packet = nullptr;

    // Video properties
    VideoProperties properties;

    // Hardware acceleration type
    std::string hwAccelType;

    // Frame properties
    int64_t pts = 0;
    Frame frame;
    // Converter for frame format
    std::unique_ptr<ffmpy::conversion::IConverter> converter;
};
} // namespace ffmpy
