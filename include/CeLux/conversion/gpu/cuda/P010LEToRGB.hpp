#pragma once

#include "BaseConverter.hpp"
#include "Frame.hpp"
extern "C"
{
    /**
     * @brief Launcher function to convert P010LE to RGB48 on the GPU.
     *
     * @param srcY       Pointer to the Y plane data.
     * @param srcPitchY  Stride (in bytes) of the Y plane.
     * @param srcUV      Pointer to the interleaved UV plane data.
     * @param srcPitchUV Stride (in bytes) of the UV plane.
     * @param dstRGB     Pointer to the output RGB buffer.
     * @param dstPitchRGB Stride (in bytes) of the RGB buffer.
     * @param width      Width of the frame.
     * @param height     Height of the frame.
     * @param stream     CUDA stream for asynchronous execution.
     */
    void P010LEToRGB48_Launcher(const uint16_t* srcY, int srcPitchY,
                                const uint16_t* srcUV, int srcPitchUV, uint16_t* dstRGB,
                                int dstPitchRGB, int width, int height,
                                cudaStream_t stream);
};

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
    ~P010LEToRGB();

    void convert(celux::Frame& frame, void* buffer) override;
};

// Template Definitions

inline P010LEToRGB::P010LEToRGB() : ConverterBase()
{
    CELUX_INFO("Creating P010LE to RGB converter (CUDA)");
}


inline P010LEToRGB::~P010LEToRGB()
{
}

/**
 * Frame is allocated with pixel format P010LE (YUV 4:2:0 10-bit planar) (already on
 * GPU) void* is originally created via torch::empty({height, width, 3},
 * torch::kFloat16).data_ptr()
 *
 * Need to ACCURATELY convert (on the gpu) the P010LE frame to RGB48 (10 bit rgb).
 * I know that most NPP and other libraries only handle 10 bit, but we can probably use
 * that with the 10 bit data?
 */
inline void P010LEToRGB::convert(celux::Frame& frame, void* buffer)
{
    CELUX_TRACE("Converting P010LE to RGB on GPU");

    // Input P010LE YUV data (16-bit per component)
    const uint16_t* pSrcY = reinterpret_cast<const uint16_t*>(frame.getData(0));
    const uint16_t* pSrcUV = reinterpret_cast<const uint16_t*>(frame.getData(1));
    int rSrcStepY = frame.getLineSize(0);  // Y plane stride in bytes
    int rSrcStepUV = frame.getLineSize(1); // UV plane stride in bytes

    // Output RGB buffer, assuming 16-bit RGB (HWC format)
    uint16_t* pDst = static_cast<uint16_t*>(buffer);
    int nDstStep = frame.getWidth() * 3 * sizeof(uint16_t); // Output stride in bytes

    int width = frame.getWidth();
    int height = frame.getHeight();

    // Call the CUDA launcher
    P010LEToRGB48_Launcher(pSrcY, rSrcStepY, pSrcUV, rSrcStepUV, pDst, nDstStep, width,
                           height, conversionStream);

    this->synchronize();
}

} // namespace cuda
} // namespace gpu
} // namespace conversion
} // namespace celux
