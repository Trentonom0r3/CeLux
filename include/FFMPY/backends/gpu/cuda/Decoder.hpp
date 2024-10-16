// CUDA Decoder.hpp
#pragma once

#include "backends/Decoder.hpp"

extern "C"
{
    #include <libavutil/hwcontext_cuda.h>
 }

namespace ffmpy::backends::gpu::cuda
{
class Decoder : public ffmpy::Decoder
{
  public:
    Decoder(const std::string& filePath,
            std::unique_ptr<ffmpy::conversion::IConverter> converter = nullptr)
        : ffmpy::Decoder(std::move(converter))
    {
        initialize(filePath);
    }

    ~Decoder() override;

  protected:
    void initHWAccel() override;
    void initCodecContext(const AVCodec* codec) override;

    static enum AVPixelFormat getHWFormat(AVCodecContext* ctx,
                                          const enum AVPixelFormat* pix_fmts);
};

} // namespace ffmpy::backends::gpu::cuda
