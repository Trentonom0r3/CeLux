// NV12ToRGB.hpp

#pragma once

#include "BaseConverter.hpp"
#include "Frame.hpp"

// Helper function to check if the frame is a hardware frame
inline bool is_hw_frame(const AVFrame* frame)
{
    return frame->hw_frames_ctx != nullptr;
}


namespace celux
{
namespace conversion
{
namespace gpu
{
namespace cuda
{

class NV12ToRGB : public ConverterBase
{
  public:
    NV12ToRGB();
    ~NV12ToRGB();

    void convert(celux::Frame& frame, void* buffer) override;
};

// Template Definitions

inline NV12ToRGB::NV12ToRGB() : ConverterBase()
{
    CELUX_INFO("Creating NV12 to RGB converter (CUDA)");
}

inline NV12ToRGB::~NV12ToRGB()
{
}

inline void NV12ToRGB::convert(celux::Frame& frame, void* buffer)
{
    if (!is_hw_frame(frame.get()))
    {
        CELUX_ERROR("Frame is not a hardware frame");
		throw std::runtime_error("Frame is not a hardware frame");
	}
    CELUX_TRACE("Converting NV12 to RGB on GPU");
    const Npp8u* pSrc[2] = {frame.getData(0), frame.getData(1)};
    int rSrcStep = frame.getLineSize(0); // Y plane stride

    Npp8u* pDst = static_cast<Npp8u*>(buffer);
    int nDstStep = frame.getWidth() * 3;

    NppiSize oSizeROI = {frame.getWidth(), frame.getHeight()};

    NppStatus status = nppiNV12ToRGB_8u_P2C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep,
                                                  oSizeROI, nppStreamContext);

    if (status != NPP_SUCCESS)
    {
		CELUX_ERROR("Failed to convert NV12 to RGB");
		throw std::runtime_error("Failed to convert NV12 to RGB");
	}

    this->synchronize();
}

} // namespace cuda
} // namespace gpu
} // namespace conversion
} // namespace celux
