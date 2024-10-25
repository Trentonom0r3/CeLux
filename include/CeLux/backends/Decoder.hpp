// Decoder.hpp
#pragma once

#include "CxException.hpp"
#include <Conversion.hpp>
#include <Frame.hpp>

namespace celux
{

class Decoder
{
  public:
    struct VideoProperties
    {
        int width;
        int height;
        double fps;
        double duration;
        double totalFrames;
        AVPixelFormat pixelFormat;
        bool hasAudio;
        int bitDepth;
        std::string codec;
    };
    Decoder() = default;
    // Constructor
    Decoder(std::optional<torch::Stream> stream);

    // Destructor
    virtual ~Decoder();

    // Deleted copy constructor and assignment operator
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    // Move constructor and assignment operator
    Decoder(Decoder&&) noexcept;
    Decoder& operator=(Decoder&&) noexcept;

    // Core methods
    virtual bool decodeNextFrame(void* buffer);
    virtual bool seek(double timestamp);
    virtual VideoProperties getVideoProperties() const;
    virtual bool isOpen() const;
    virtual void close();
    virtual std::vector<std::string> listSupportedDecoders() const;
    AVCodecContext* getCtx();

    // getter for bit depth
    int getBitDepth() const;

  protected:
    // Initialization method
    void initialize(const std::string& filePath);
    bool isHardwareAccelerated(const AVCodec* codec);
    void setProperties();
    // Virtual methods for customization
    virtual void openFile(const std::string& filePath);
    virtual void initHWAccel(); // Default does nothing
    virtual void findVideoStream();
    virtual void initCodecContext();
    virtual int64_t convertTimestamp(double timestamp) const;
    void populateProperties();
    void setFormatFromBitDepth();
    // Deleters and smart pointers
    struct AVFormatContextDeleter
    {
        void operator()(AVFormatContext* ctx) const
        {
            avformat_close_input(&ctx);
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

    struct AVPacketDeleter
    {
        void operator()(AVPacket* pkt) const
        {
            av_packet_free(&pkt);
        }
    };

    using AVFormatContextPtr = std::unique_ptr<AVFormatContext, AVFormatContextDeleter>;
    using AVCodecContextPtr = std::unique_ptr<AVCodecContext, AVCodecContextDeleter>;
    using AVBufferRefPtr = std::unique_ptr<AVBufferRef, AVBufferRefDeleter>;
    using AVPacketPtr = std::unique_ptr<AVPacket, AVPacketDeleter>;
    void set_sw_pix_fmt(AVCodecContextPtr& codecCtx, int bitDepth);
    // Member variables
    AVFormatContextPtr formatCtx;
    AVCodecContextPtr codecCtx;
    AVPacketPtr pkt;
    int videoStreamIndex;
    VideoProperties properties;
    Frame frame;
    bool isHwAccel;
    std::unique_ptr<celux::conversion::IConverter> converter;
    AVBufferRefPtr hwDeviceCtx; // For hardware acceleration
    AVBufferRefPtr hwFramesCtx; // For hardware acceleration
    std::optional<torch::Stream> decoderStream;
};
} // namespace celux
