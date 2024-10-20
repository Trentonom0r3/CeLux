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
        hwAccelType = hwType;
        this->initialize(outputPath, props);
    }



  protected:
    void initHWAccel() override;
    enum AVPixelFormat getHWFormat(AVCodecContext* ctx,
                                   const enum AVPixelFormat* pix_fmts) override;
    void configureCodecContext(const AVCodec* codec,
                               const VideoProperties& props) override;

}; // class Encoder
} // namespace celux::backends::gpu::cuda

