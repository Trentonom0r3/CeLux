#pragma once

#include "error/CxException.hpp"
#include <Conversion.hpp>
#include <Frame.hpp>

namespace celux
{

class Decoder
{
  public:

    struct VideoProperties
    {
        std::string codec;
        int width;
        int height;
        double fps;
        double duration;
        int totalFrames;
        AVPixelFormat pixelFormat;
        bool hasAudio;
        int bitDepth;
        double aspectRatio;
        int audioBitrate;
        int audioChannels;
        int audioSampleRate;
        std::string audioCodec;
        double min_fps;
        double max_fps;
    };

    Decoder() = default;
    Decoder(int numThreads);
    bool seekToNearestKeyframe(double timestamp);
    virtual ~Decoder();

    // Deleted copy constructor and assignment operator
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    Decoder(Decoder&&) noexcept;
    Decoder& operator=(Decoder&&) noexcept;
    bool seekFrame(int frameIndex);
    virtual bool decodeNextFrame(void* buffer, double* frame_timestamp = nullptr);
    virtual bool seek(double timestamp);
    virtual VideoProperties getVideoProperties() const;
    virtual bool isOpen() const;
    virtual void close();
    virtual std::vector<std::string> listSupportedDecoders() const;
    AVCodecContext* getCtx();
    int getBitDepth() const;
    bool extractAudioToFile(const std::string& outputFilePath);
    torch::Tensor getAudioTensor();

  protected:
    void initialize(const std::string& filePath);
    void setProperties();
    virtual void openFile(const std::string& filePath);
    virtual void findVideoStream();
    virtual void initCodecContext();
    virtual int64_t convertTimestamp(double timestamp) const;

    double getFrameTimestamp(AVFrame* frame);


    AVFilterGraphPtr filter_graph_;
    AVFilterContext* buffersrc_ctx_;
    AVFilterContext* buffersink_ctx_;
    AVFormatContextPtr formatCtx;
    AVCodecContextPtr codecCtx;
    AVPacketPtr pkt;
    int videoStreamIndex;
    VideoProperties properties;
    Frame frame; // custom wrapper around AVframe, handles mem, most ops w/ it.
    std::unique_ptr<celux::conversion::IConverter> converter;
    int numThreads;

    int audioStreamIndex = -1;
    AVCodecContextPtr audioCodecCtx;
    Frame audioFrame;
    AVPacketPtr audioPkt;
    SwrContextPtr swrCtx;

    bool initializeAudio();
    void closeAudio();
};
} // namespace celux
