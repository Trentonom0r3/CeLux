// Decoder.hpp
#pragma once

#include "Frame.hpp" // RAII wrapper for AVFrame

namespace ffmpy
{
/**
 * @class Decoder
 * @brief A class for decoding video files using FFmpeg.
 */
class Decoder
{
  public:
    /**
     * @struct VideoProperties
     * @brief Stores properties of the video.
     */
    struct VideoProperties
    {
        int width;                 ///< Width of the video in pixels.
        int height;                ///< Height of the video in pixels.
        double fps;                ///< Frames per second of the video.
        double duration;           ///< Duration of the video in seconds.
        int totalFrames;           ///< Total number of frames in the video.
        AVPixelFormat pixelFormat; ///< Pixel format of the video.
    };

    /**
     * @brief Constructs a Decoder object.
     *
     * @param filePath Path to the video file.
     * @param useHardware Whether to use hardware acceleration for decoding (default:
     * true).
     * @param hwType Type of hardware acceleration to use (default: "cuda").
     */
    Decoder(const std::string& filePath, bool useHardware = true,
            const std::string& hwType = "cuda");

    /**
     * @brief Destructor for the Decoder object.
     */
    virtual ~Decoder();

    /**
     * @brief Deleted copy constructor to disable copy semantics.
     */
    Decoder(const Decoder&) = delete;

    /**
     * @brief Deleted copy assignment operator to disable copy semantics.
     */
    Decoder& operator=(const Decoder&) = delete;

    /**
     * @brief Move constructor to enable move semantics.
     */
    Decoder(Decoder&&) noexcept;

    /**
     * @brief Move assignment operator to enable move semantics.
     */
    Decoder& operator=(Decoder&&) noexcept;

    /**
     * @brief Decodes the next frame in the video.
     *
     * @param frame A reference to a Frame object to store the decoded frame.
     * @return True if the frame was successfully decoded, false otherwise.
     */
    bool decodeNextFrame(Frame& frame);

    /**
     * @brief Seeks to a specific timestamp in the video.
     *
     * @param timestamp The timestamp in seconds to seek to.
     * @return True if the seek was successful, false otherwise.
     */
    bool seek(double timestamp);

    /**
     * @brief Gets the properties of the video.
     *
     * @return A VideoProperties struct containing the video properties.
     */
    VideoProperties getVideoProperties() const;

    /**
     * @brief Checks if the video file is open.
     *
     * @return True if the video file is open, false otherwise.
     */
    bool isOpen() const;

    /**
     * @brief Closes the video file.
     */
    void close();

    /**
     * @brief Lists the supported video decoders.
     *
     * @return A vector of strings containing the names of the supported decoders.
     */
    std::vector<std::string> listSupportedDecoders() const;

    /**
     * @brief Gets the codec context.
     *
     * @return A pointer to the AVCodecContext.
     */
    AVCodecContext* getCtx()
    {
        return codecCtx.get();
    }

  private:
    /**
     * @brief Opens the video file and initializes decoding contexts.
     *
     * @param filePath Path to the video file.
     * @param useHardware Whether to use hardware acceleration for decoding.
     * @param hwType Type of hardware acceleration to use.
     */
    void openFile(const std::string& filePath, bool useHardware,
                  const std::string& hwType);

    /**
     * @brief Initializes hardware acceleration.
     *
     * @param hwType Type of hardware acceleration to use.
     */
    void initHWAccel(const std::string& hwType);

    /**
     * @brief Finds the video stream in the input file.
     */
    void findVideoStream();

    /**
     * @brief Initializes the codec context for decoding.
     *
     * @param codec A pointer to the AVCodec to be used for decoding.
     */
    void initCodecContext(const AVCodec* codec);

    /**
     * @brief Converts a timestamp from seconds to the appropriate format for seeking.
     *
     * @param timestamp The timestamp in seconds.
     * @return The converted timestamp.
     */
    int64_t convertTimestamp(double timestamp) const;

    /**
     * @brief Gets the hardware pixel format for the codec context.
     *
     * @param ctx A pointer to the AVCodecContext.
     * @param pix_fmts A pointer to the list of supported pixel formats.
     * @return The selected hardware pixel format.
     */
    static enum AVPixelFormat getHWFormat(AVCodecContext* ctx,
                                          const enum AVPixelFormat* pix_fmts);

    /**
     * @struct AVFormatContextDeleter
     * @brief Custom deleter for AVFormatContext.
     */
    struct AVFormatContextDeleter
    {
        /**
         * @brief Deletes the AVFormatContext by closing the input.
         *
         * @param ctx A pointer to the AVFormatContext to be deleted.
         */
        void operator()(AVFormatContext* ctx) const
        {
            avformat_close_input(&ctx);
        }
    };

    /**
     * @struct AVCodecContextDeleter
     * @brief Custom deleter for AVCodecContext.
     */
    struct AVCodecContextDeleter
    {
        /**
         * @brief Deletes the AVCodecContext by freeing the context.
         *
         * @param ctx A pointer to the AVCodecContext to be deleted.
         */
        void operator()(AVCodecContext* ctx) const
        {
            avcodec_free_context(&ctx);
        }
    };

    /**
     * @struct AVBufferRefDeleter
     * @brief Custom deleter for AVBufferRef.
     */
    struct AVBufferRefDeleter
    {
        /**
         * @brief Deletes the AVBufferRef by unreferencing the buffer.
         *
         * @param ref A pointer to the AVBufferRef to be deleted.
         */
        void operator()(AVBufferRef* ref) const
        {
            av_buffer_unref(&ref);
        }
    };
    using AVFormatContextPtr = std::unique_ptr<AVFormatContext, AVFormatContextDeleter>;

    using AVCodecContextPtr = std::unique_ptr<AVCodecContext, AVCodecContextDeleter>;

    using AVBufferRefPtr = std::unique_ptr<AVBufferRef, AVBufferRefDeleter>;
    AVFormatContextPtr formatCtx; ///< Format context for the video file
    AVCodecContextPtr codecCtx;   ///< Codec context for decoding
    AVBufferRefPtr hwDeviceCtx;   ///< Hardware device context
    AVPacket* pkt;                ///< Packet for storing compressed data
    int videoStreamIndex;         ///< Index of the video stream
    VideoProperties properties;   ///< Video properties
    std::string hwAccelType;      ///< Hardware acceleration type
};
} // namespace ffmpy
