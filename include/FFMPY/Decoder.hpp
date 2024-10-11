// Decoder.hpp
#pragma once

#include "FFException.hpp"
#include <Frame.hpp>
#include <NV12ToRGB.hpp>
#include <memory>

namespace ffmpy
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
        int totalFrames;
        AVPixelFormat pixelFormat;
        bool audio;
    };

    // Updated constructor to accept shared_ptr
    Decoder(const std::string& filePath, bool useHardware = true,
                const std::string& hwType = "cuda",
                std::unique_ptr<ffmpy::conversion::IConverter> converter = nullptr);

    AVBufferRef* getHWDeviceCtx() const;
    virtual ~Decoder();

    // Deleted copy constructor and assignment operator
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    // Move constructor and assignment operator
    Decoder(Decoder&&) noexcept;
    Decoder& operator=(Decoder&&) noexcept;

    bool decodeNextFrame(void* buffer);
    bool seek(double timestamp);
    VideoProperties getVideoProperties() const;
    bool isOpen() const;
    void close();
    std::vector<std::string> listSupportedDecoders() const;
    AVCodecContext* getCtx()
    {
        return codecCtx.get();
    }

  private:
    void openFile(const std::string& filePath, bool useHardware,
                  const std::string& hwType);
    void initHWAccel(const std::string& hwType);
    void findVideoStream();
    void initCodecContext(const AVCodec* codec);
    int64_t convertTimestamp(double timestamp) const;
    static enum AVPixelFormat getHWFormat(AVCodecContext* ctx,
                                          const enum AVPixelFormat* pix_fmts);

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

    using AVFormatContextPtr = std::unique_ptr<AVFormatContext, AVFormatContextDeleter>;
    using AVCodecContextPtr = std::unique_ptr<AVCodecContext, AVCodecContextDeleter>;
    using AVBufferRefPtr = std::unique_ptr<AVBufferRef, AVBufferRefDeleter>;

    AVFormatContextPtr formatCtx;
    AVCodecContextPtr codecCtx;
    AVBufferRefPtr hwDeviceCtx;
    AVPacket* pkt;
    int videoStreamIndex;
    VideoProperties properties;
    std::string hwAccelType;
    Frame frame;

    // Changed to shared_ptr
    std::unique_ptr<ffmpy::conversion::IConverter> converter;
};
} // namespace ffmpy
