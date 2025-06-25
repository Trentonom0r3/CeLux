#pragma once
#ifndef ENCODER_HPP
#define ENCODER_HPP

#include "error/CxException.hpp"
#include <Conversion.hpp>
#include <FilterFactory.hpp>
#include <Frame.hpp>
#include <filesystem>

namespace celux
{

class Encoder
{
  public:
    struct EncodingProperties
    {
        std::string codec;
        int width;
        int height;
        int bitRate;
        AVPixelFormat pixelFormat;
        int gopSize;
        int maxBFrames;
        int fps;
        int audioBitRate;
        int audioSampleRate;
        int audioChannels;
        std::string audioCodec;
    };

    Encoder() = default;
    Encoder(const std::string& filename, const EncodingProperties& properties);
    ~Encoder();

    void initialize();
    bool encodeFrame(const Frame& frame);
    bool encodeAudioFrame(const Frame& frame);
    void writePacket();
    void close();

    // Deleted copy constructor and assignment operator
    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;

  private:
    void initVideoStream();
    void initAudioStream();
    void openOutputFile();
    void validateCodecContainerCompatibility();
    std::string
    inferContainerFormat(const std::string& filename) const; // New helper function

    EncodingProperties properties;
    std::string filename; // Store filename for later use
    AVFormatContextPtr formatCtx;
    AVCodecContextPtr videoCodecCtx;
    AVCodecContextPtr audioCodecCtx;
    AVStream* videoStream = nullptr;
    AVStream* audioStream = nullptr;
    SwrContextPtr swrCtx;
    AVPacketPtr pkt;
};

} // namespace celux

#endif // ENCODER_HPP
