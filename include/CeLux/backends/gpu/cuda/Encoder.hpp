#pragma once

#include "backends/Encoder.hpp"

namespace celux::backends::gpu::cuda
{
class Encoder : public celux::Encoder
{
  public:
    Encoder(const std::string& outputPath, const VideoProperties& props,
            std::unique_ptr<celux::conversion::IConverter> converter = nullptr,
            const std::string& hwType = "cuda")
        : celux::Encoder(std::move(converter))
    {
        std::cout << "Initializing CUDA Encoder, in constructor\n" << std::endl;
        hwAccelType = hwType;
        this->initialize(outputPath, props);
    }

    ~Encoder() override
    {
        finalize();
    }

  protected:
    void initHWAccel() override;
    void initCodecContext(const AVCodec* codec, const VideoProperties& props) override;
    enum AVPixelFormat getHWFormat(AVCodecContext* ctx,
                                   const enum AVPixelFormat* pix_fmts) override;
};
} // namespace celux::backends::gpu::cuda
