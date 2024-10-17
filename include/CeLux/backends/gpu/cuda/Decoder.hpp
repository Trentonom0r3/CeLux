// CUDA Decoder.hpp
#pragma once

#include "backends/Decoder.hpp"

namespace celux::backends::gpu::cuda
{
class Decoder : public celux::Decoder
{
  public:
    Decoder(const std::string& filePath,
            std::unique_ptr<celux::conversion::IConverter> converter = nullptr)
        : celux::Decoder(std::move(converter))
    {
        initialize(filePath);
    }

    ~Decoder() override ;

  protected:
    void initHWAccel() override;
    void initCodecContext(const AVCodec* codec) override;

    static enum AVPixelFormat getHWFormat(AVCodecContext* ctx,
                                          const enum AVPixelFormat* pix_fmts);
};

} // namespace celux::backends::gpu::cuda
