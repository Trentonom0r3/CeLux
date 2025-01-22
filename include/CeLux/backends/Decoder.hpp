// Decoder.hpp
#pragma once

#include "CxException.hpp"
#include <Conversion.hpp>
#include <FilterFactory.hpp>
#include <Frame.hpp>
#include <backends/cpu/Remuxer.hpp>

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
        double aspectRatio;     // New property
        int audioBitrate;       // New property
        int audioChannels;      // New property
        int audioSampleRate;    // New property
        std::string audioCodec; // New property
        double min_fps;         // New property for minimum fps
        double max_fps;         // New property for maximum fps
    };

    Decoder() = default;
    // Constructor
    Decoder(int numThreads, std::vector<std::shared_ptr<FilterBase>> filters);
    bool seekToNearestKeyframe(double timestamp);
    // Destructor
    virtual ~Decoder();

    // Deleted copy constructor and assignment operator
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;
    /**
     * @brief Adds a filter to the decoder's filter pipeline.
     *
     * @param filter Shared pointer to a Filter instance.
     */
    void addFilter(const std::unique_ptr<FilterBase>& filter);
    // Move constructor and assignment operator
    Decoder(Decoder&&) noexcept;
    Decoder& operator=(Decoder&&) noexcept;

    /**
     * @brief Decode the next frame and store it in the provided buffer.
     *
     * @param buffer Pointer to the buffer where the frame data will be stored.
     * @param frame_timestamp Optional pointer to a double where the frame's timestamp
     * will be stored.
     * @return true if a frame was successfully decoded, false otherwise.
     */
    virtual bool decodeNextFrame(void* buffer, double* frame_timestamp = nullptr);

    virtual bool seek(double timestamp);
    virtual VideoProperties getVideoProperties() const;
    virtual bool isOpen() const;
    virtual void close();
    virtual std::vector<std::string> listSupportedDecoders() const;
    AVCodecContext* getCtx();
    /**
     * @brief Seek to a precise timestamp by decoding frames after keyframe seeking.
     *
     * @param timestamp Target timestamp in seconds.
     * @return true if successful, false otherwise.
     */
    bool seekToPreciseTimestamp(double timestamp);

    // getter for bit depth
    int getBitDepth() const;

  protected:
    // Initialization method
    void initialize(const std::string& filePath);
    void setProperties();
    // Virtual methods for customization
    virtual void openFile(const std::string& filePath);
    virtual void findVideoStream();
    virtual void initCodecContext();
    virtual int64_t convertTimestamp(double timestamp) const;
    void populateProperties();
    void setFormatFromBitDepth();

    /**
     * @brief Get the timestamp of the frame in seconds.
     *
     * @param frame Pointer to the AVFrame.
     * @return double Timestamp in seconds.
     */
    double getFrameTimestamp(AVFrame* frame);

    std::vector<std::shared_ptr<FilterBase>> filters_;

    /**
     * @brief Initializes the filter graph based on the added filters.
     *
     * @return true if successful.
     * @return false otherwise.
     */
    bool initFilterGraph();

    void set_sw_pix_fmt(AVCodecContextPtr& codecCtx, int bitDepth);

    // Member variables

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
    Remuxer remuxer_; // Keep a member to store the remuxer
};
} // namespace celux
