#pragma once

#include "CxException.hpp"
#include <Conversion.hpp>
#include <FilterFactory.hpp>
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
        bool isRawVideo; // ✅ New: Identifies raw video formats
        bool needsRemux; // ✅ New: Flag if remuxing is needed
    };

    Decoder() = default;
    Decoder(int numThreads, std::vector<std::shared_ptr<FilterBase>> filters);
    bool seekToNearestKeyframe(double timestamp);
    virtual ~Decoder();

    // Deleted copy constructor and assignment operator
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    void addFilter(const std::unique_ptr<FilterBase>& filter);
    Decoder(Decoder&&) noexcept;
    Decoder& operator=(Decoder&&) noexcept;

    virtual bool decodeNextFrame(void* buffer, double* frame_timestamp = nullptr);
    virtual bool seek(double timestamp);
    virtual VideoProperties getVideoProperties() const;
    virtual bool isOpen() const;
    virtual void close();
    virtual std::vector<std::string> listSupportedDecoders() const;
    AVCodecContext* getCtx();
    bool seekToPreciseTimestamp(double timestamp);
    int getBitDepth() const;

  protected:
    void initialize(const std::string& filePath);
    void setProperties();
    virtual void openFile(const std::string& filePath);
    virtual void findVideoStream();
    virtual void initCodecContext();
    virtual int64_t convertTimestamp(double timestamp) const;
    void populateProperties();
    void setFormatFromBitDepth();
    double getFrameTimestamp(AVFrame* frame);

    std::vector<std::shared_ptr<FilterBase>> filters_;

    bool initFilterGraph();
    void set_sw_pix_fmt(AVCodecContextPtr& codecCtx, int bitDepth);

    // ✅ New Methods for Handling Raw Video & Remuxing
    bool isRawFormat(const std::string& filePath);
    std::string remuxToSupportedFormat(const std::string& inputPath);

    AVFilterGraphPtr filter_graph_;
    AVFilterContext* buffersrc_ctx_;
    AVFilterContext* buffersink_ctx_;
    AVFormatContextPtr formatCtx;
    AVCodecContextPtr codecCtx;
    AVPacketPtr pkt;
    int videoStreamIndex;
    VideoProperties properties;
    Frame frame;
    std::unique_ptr<celux::conversion::IConverter> converter;
    int numThreads;
};
} // namespace celux
