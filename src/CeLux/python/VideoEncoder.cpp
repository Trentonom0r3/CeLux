#include "python/VideoEncoder.hpp"
#include <filesystem>
#include <stdexcept>
#include <FilterFactory.hpp>
#include <Factory.hpp>

namespace fs = std::filesystem;

namespace celux
{

VideoEncoder::VideoEncoder(const std::string& filename,
                           std::optional<std::string> codec, std::optional<int> width,
                           std::optional<int> height, std::optional<int> bitRate,
                           std::optional<int> fps, std::optional<int> audioBitRate,
                           std::optional<int> audioSampleRate,
                           std::optional<int> audioChannels,
                           std::optional<std::string> audioCodec)
{
    auto properties = inferEncodingProperties(filename, codec, width, height, bitRate,
                                              fps, audioBitRate, audioSampleRate,
                                              audioChannels, audioCodec);
    this->width = properties.width;
    this->height = properties.height;

    encoder = std::make_unique<celux::Encoder>(filename, properties);
}

celux::Encoder::EncodingProperties VideoEncoder::inferEncodingProperties(
    const std::string& filename, std::optional<std::string> codec,
    std::optional<int> width, std::optional<int> height, std::optional<int> bitRate,
    std::optional<int> fps, std::optional<int> audioBitRate,
    std::optional<int> audioSampleRate, std::optional<int> audioChannels,
    std::optional<std::string> audioCodec)
{

    // Default codec selection
    props.codec = codec.value_or("libx264");
    props.audioCodec = audioCodec.value_or("aac");

    // Infer resolution
    props.width = width.value_or(1920);
    props.height = height.value_or(1080);

    // Bitrate defaults
    props.bitRate = bitRate.value_or(4000000);          // 4 Mbps
    props.audioBitRate = audioBitRate.value_or(192000); // 192 kbps

    // FPS default
    props.fps = fps.value_or(30);
    props.gopSize = 60;   // GOP of 60 frames
    props.maxBFrames = 2; // Max 2 B-frames

    // Audio properties
    props.audioSampleRate = audioSampleRate.value_or(44100);
    props.audioChannels = audioChannels.value_or(2); // Default to stereo

    // Set pixel format (will be determined in `encodeFrame`)
    props.pixelFormat = AV_PIX_FMT_YUV420P;

    return props;
}

void VideoEncoder::encodeFrame(const torch::Tensor& frame)
{
    if (!encoder)
    {
        throw std::runtime_error("Encoder is not initialized");
    }
    // Convert from RGB to YUV420P (or any required format)
    // Pseudocode for how to do it explicitly:
    celux::Frame convertedFrame; // 1) default constructor
    convertedFrame.get()->format = AV_PIX_FMT_YUV420P;
    convertedFrame.get()->width = width;   // from your encoder->width
    convertedFrame.get()->height = height; // from your encoder->height
    convertedFrame.allocateBuffer(32);     // 2) allocate actual data planes

   // std::unique_ptr<celux::conversion::IConverter> converter =
    //    celux::conversion::createConverter(inputPixelFormat, AV_PIX_FMT_YUV420P);
    if (!converter){
        converter = celux::Factory::createEncodeConverter(torch::kCPU, props.pixelFormat);
    //converter->convertFrame(convertedFrame, frame.data_ptr()); // Use void* data_ptr()
    }
    converter->convert(convertedFrame, frame.contiguous().data_ptr());
    // Encode the converted frame
    encoder->encodeFrame(convertedFrame);
}

void VideoEncoder::encodeAudioFrame(const torch::Tensor& audio)
{
    if (!encoder)
    {
        throw std::runtime_error("Encoder is not initialized");
    }

    if (audio.scalar_type() != torch::kUInt8 && audio.scalar_type() != torch::kUInt16)
    {
        throw std::runtime_error("Input tensor must be uint8 or uint16");
    }

    celux::Frame encodedAudio;
   // encodedAudio.fillData(audio.data_ptr(), audio.numel(), 0);
    encoder->encodeAudioFrame(encodedAudio);
}

void VideoEncoder::close()
{
    if (encoder)
    {
        encoder->close();
        encoder.reset();
    }
}

VideoEncoder::~VideoEncoder()
{
}

} // namespace celux
