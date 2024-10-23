// RGBToYUV420P.hpp

#pragma once

#include "Frame.hpp"
#include "BaseConverter.hpp"

namespace celux
{
namespace conversion
{
namespace gpu
{
namespace cuda
{
    //TODO RENAME TO RGBToYUV420
class RGBToYUV420P : public ConverterBase
{
  public:
    RGBToYUV420P();
    RGBToYUV420P(cudaStream_t stream);
    ~RGBToYUV420P();

    void convert(celux::Frame& frame, void* buffer) override;
};

// Template Definitions

inline RGBToYUV420P::RGBToYUV420P() : ConverterBase()
{
}

inline RGBToYUV420P::RGBToYUV420P(cudaStream_t stream) : ConverterBase(stream)
{
}

inline RGBToYUV420P::~RGBToYUV420P()
{
}

inline void RGBToYUV420P::convert(celux::Frame& frame, void* buffer)
{
    Npp8u* pSrcRGB = static_cast<Npp8u*>(buffer);
    CELUX_DEBUG("RGBToYUV420P: Converting RGB to YUV420P");
    CELUX_DEBUG("Frame Format: {}", frame.getPixelFormatString());
    // Array of pointers for Y, U, and V planes
    Npp8u* pDst[3];
    pDst[0] = frame.getData(0); // Y plane
    pDst[1] = frame.getData(1); // U plane
    pDst[2] = frame.getData(2); // V plane

    // Array of strides for Y, U, and V planes
    int nDstStep[3];
    nDstStep[0] = frame.getLineSize(0); // Y stride
    nDstStep[1] = frame.getLineSize(1); // U stride
    nDstStep[2] = frame.getLineSize(2); // V stride

    int nSrcStep = frame.getWidth() * 3; // Assuming RGB stride is width * 3
    NppiSize oSizeROI = {frame.getWidth(), frame.getHeight()};

    // Perform RGB to YUV420 conversion using NPP
    NppStatus status =
        nppiRGBToYUV420_8u_C3P3R_Ctx(pSrcRGB, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamContext);

}


} // namespace cuda
} // namespace gpu
} // namespace conversion
} // namespace celux
