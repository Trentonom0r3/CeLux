#pragma once

#include "backends/Encoder.hpp"

namespace celux::backends::gpu::cuda
{

class Encoder : public celux::Encoder
{
  public:
    Encoder(const std::string& outputPath, int width, int height, double fps,
            celux::EncodingFormats format, const std::string& codecName,
            std::optional<torch::Stream> stream)
        :
        celux::Encoder(outputPath, width, height, fps, format, codecName, stream)
    {
        hwAccelType = "cuda";
        this->initialize();
    }



  protected:
    void initHWAccel() override;
    enum AVPixelFormat getHWFormat(AVCodecContext* ctx,
                                   const enum AVPixelFormat* pix_fmts) override;
    void configureCodecContext(const AVCodec* codec) override;

}; // class Encoder
} // namespace celux::backends::gpu::cuda

