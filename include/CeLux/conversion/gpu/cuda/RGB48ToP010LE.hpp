// RGBToP010LE.hpp

#pragma once

#include "BaseConverter.hpp"
#include "Frame.hpp"
#include <cstdint>
#include <cuda_runtime.h>

namespace celux
{
namespace conversion
{
namespace gpu
{
namespace cuda
{

extern "C"
{
    /**
     * @brief Launcher function to convert RGB48LE to YUV420P10LE on the GPU.
     *
     * @param srcRGB        Pointer to the input RGB data.
     * @param srcPitchRGB   Stride (in bytes) of the RGB data.
     * @param dstY          Pointer to the output Y plane data.
     * @param dstPitchY     Stride (in bytes) of the Y plane.
     * @param dstUV         Pointer to the output interleaved UV plane data.
     * @param dstPitchUV    Stride (in bytes) of the UV plane.
     * @param width         Width of the frame.
     * @param height        Height of the frame.
     * @param stream        CUDA stream for asynchronous execution.
     */
    void RGBToP010LE_Launcher(const uint16_t* srcRGB, int srcPitchRGB, uint16_t* dstY,
                              int dstPitchY, uint16_t* dstUV, int dstPitchUV, int width,
                              int height, cudaStream_t stream);
}
/**
 * @brief Class responsible for converting RGB48LE frames to YUV420P10LE.
 */
class RGBToP010LE : public ConverterBase
{
  public:
    /**
     * @brief Default constructor.
     */
    RGBToP010LE();

    /**
     * @brief Constructor with CUDA stream.
     *
     * @param stream CUDA stream for asynchronous execution.
     */
    RGBToP010LE(cudaStream_t stream);

    /**
     * @brief Destructor.
     */
    ~RGBToP010LE();

    /**
     * @brief Converts an RGB48LE frame to YUV420P10LE format on the GPU.
     *
     * @param frame  Reference to the input RGB48LE frame.
     * @param buffer Pointer to the pre-allocated output YUV420P10LE buffer.
     *
     * @throws std::runtime_error if the conversion fails.
     */
    void convert(celux::Frame& frame, void* buffer) override;
};

// Inline Definitions

inline RGBToP010LE::RGBToP010LE() : ConverterBase()
{
}

inline RGBToP010LE::RGBToP010LE(cudaStream_t stream) : ConverterBase(stream)
{
}

inline RGBToP010LE::~RGBToP010LE()
{
}

inline void RGBToP010LE::convert(celux::Frame& frame, void* buffer)
{
    CELUX_TRACE("Converting RGB48LE to YUV420P10LE on GPU");

    // Input RGB48LE data (16-bit per component)
    const uint16_t* pSrcRGB = reinterpret_cast<const uint16_t*>(frame.getData(0));
    int srcPitchRGB = frame.getLineSize(0); // RGB stride in bytes

    // Output YUV420P10LE buffer, assuming planar format with Y and interleaved UV
    // planes
    uint16_t* pDstY = static_cast<uint16_t*>(buffer);
    uint16_t* pDstUV = pDstY + frame.getWidth() * frame.getHeight(); // Y followed by UV

    int dstPitchY = frame.getWidth() * sizeof(uint16_t); // Y plane stride in bytes
    int dstPitchUV = (frame.getWidth() / 2) * 2 *
                     sizeof(uint16_t); // UV plane stride in bytes (interleaved U and V)

    int width = frame.getWidth();
    int height = frame.getHeight();

    // Use the CUDA stream from the base class
    cudaStream_t stream = this->getStream();
    CELUX_INFO("Beginning RGB48 Conversion");
    // Call the CUDA launcher
    RGBToP010LE_Launcher(pSrcRGB, srcPitchRGB, pDstY, dstPitchY, pDstUV, dstPitchUV,
                         width, height, stream);

    // Error handling is performed within the launcher
}

} // namespace cuda
} // namespace gpu
} // namespace conversion
} // namespace celux
