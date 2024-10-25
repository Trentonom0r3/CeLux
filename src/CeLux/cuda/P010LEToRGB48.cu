// P010LEToRGB.cu

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// CUDA kernel for converting P010LE to RGB48
__global__ void P010LEToRGB48Kernel(const uint16_t* srcY, int srcPitchY,
                                    const uint16_t* srcUV, int srcPitchUV,
                                    uint16_t* dstRGB, int dstPitchRGB, int width,
                                    int height)
{
    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (x >= width || y >= height)
        return;

    // Index calculations
    int idxY = y * (srcPitchY / sizeof(uint16_t)) + x;
    uint16_t Y = srcY[idxY];

    int uv_x = x / 2;
    int uv_y = y / 2;
    int idxUV = (uv_y * (srcPitchUV / sizeof(uint16_t))) + (uv_x * 2);
    uint16_t U = srcUV[idxUV];
    uint16_t V = srcUV[idxUV + 1];

    // Extract 10-bit values by shifting
    Y >>= 6;
    U >>= 6;
    V >>= 6;

    // Normalize YUV values
    float Yf = ((float)Y - 64.0f) / 876.0f;  // Y range: [64, 940]
    float Uf = ((float)U - 512.0f) / 448.0f; // U range: [64, 960]
    float Vf = ((float)V - 512.0f) / 448.0f; // V range: [64, 960]

    // YUV to RGB conversion (BT.709)
    float Rf = Yf + 1.5748f * Vf;
    float Gf = Yf - 0.1873f * Uf - 0.4681f * Vf;
    float Bf = Yf + 1.8556f * Uf;

    // Clamp to [0.0, 1.0]
    Rf = fminf(fmaxf(Rf, 0.0f), 1.0f);
    Gf = fminf(fmaxf(Gf, 0.0f), 1.0f);
    Bf = fminf(fmaxf(Bf, 0.0f), 1.0f);

    // Scale to 16-bit RGB
    uint16_t R = static_cast<uint16_t>(Rf * 65535.0f + 0.5f);
    uint16_t G = static_cast<uint16_t>(Gf * 65535.0f + 0.5f);
    uint16_t B = static_cast<uint16_t>(Bf * 65535.0f + 0.5f);

    // Output index calculations
    int idxRGB = (y * (dstPitchRGB / sizeof(uint16_t))) + (x * 3);
    dstRGB[idxRGB] = R;
    dstRGB[idxRGB + 1] = G;
    dstRGB[idxRGB + 2] = B;
}

// Launcher function with C linkage
extern "C" void P010LEToRGB48_Launcher(const uint16_t* srcY, int srcPitchY,
                                       const uint16_t* srcUV, int srcPitchUV,
                                       uint16_t* dstRGB, int dstPitchRGB, int width,
                                       int height, cudaStream_t stream)
{
    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    P010LEToRGB48Kernel<<<gridSize, blockSize, 0, stream>>>(
        srcY, srcPitchY, srcUV, srcPitchUV, dstRGB, dstPitchRGB, width, height);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    // Optionally synchronize the stream to ensure completion
    // cudaStreamSynchronize(stream);
}
