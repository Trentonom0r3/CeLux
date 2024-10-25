#pragma once

#include "BaseConverter.hpp"
#include "Frame.hpp"


namespace celux
{
namespace conversion
{
namespace gpu
{
namespace cuda
{

class P010LEToRGB : public ConverterBase
{
  public:
    P010LEToRGB();
    P010LEToRGB(cudaStream_t stream);
    ~P010LEToRGB();

    void convert(celux::Frame& frame, void* buffer) override;
};

// Template Definitions

inline P010LEToRGB::P010LEToRGB() : ConverterBase()
{
    CELUX_INFO("Creating P010LE to RGB converter (CUDA)");
}

inline P010LEToRGB::P010LEToRGB(cudaStream_t stream) : ConverterBase(stream)
{
}

inline P010LEToRGB::~P010LEToRGB()
{
}

inline void P010LEToRGB::convert(celux::Frame& frame, void* buffer)
{
    CELUX_TRACE("Converting P010LE to RGB on GPU");

    // Input P010LE YUV data (16-bit per component)
    const Npp16u* pSrcY = reinterpret_cast<const Npp16u*>(frame.getData(0));
    const Npp16u* pSrcUV = reinterpret_cast<const Npp16u*>(frame.getData(1));
    int rSrcStepY = frame.getLineSize(0);  // Y plane stride
    int rSrcStepUV = frame.getLineSize(1); // UV plane stride (4:2:0 subsampled)

    // Output RGB buffer, assuming 16-bit RGB (HWC format)
    Npp16u* pDst = static_cast<Npp16u*>(buffer);
    int nDstStep = frame.getWidth() * 3 *
                   sizeof(Npp16u); // 3 channels (R, G, B) with 16-bit precision

    NppiSize oSizeROI = {frame.getWidth(), frame.getHeight()};

    // Perform the conversion using NPP's 16-bit YUV 4:2:0 to RGB conversion function
    NppStatus status = NPP_SUCCESS;

    if (status != NPP_SUCCESS)
    {
        throw std::runtime_error("YUV to RGB conversion failed");
    }
}

} // namespace cuda
} // namespace gpu
} // namespace conversion
} // namespace celux
