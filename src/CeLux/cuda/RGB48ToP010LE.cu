// RGBToP010LE.cu
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <algorithm>

// CUDA kernel for converting RGB48LE to YUV420P10LE
__global__ void RGBToP010LEKernel(const uint16_t* srcRGB, int srcPitchRGB,
                                  uint16_t* dstY, int dstPitchY, uint16_t* dstUV,
                                  int dstPitchUV, int width, int height)
{
    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure we are within the image bounds
    if (x >= width || y >= height)
        return;

    // Index calculations for RGB (HWC format: R, G, B)
    int idxRGB = y * (srcPitchRGB / sizeof(uint16_t)) + x * 3;

    uint16_t R = srcRGB[idxRGB];
    uint16_t G = srcRGB[idxRGB + 1];
    uint16_t B = srcRGB[idxRGB + 2];

    // Normalize RGB values from [0, 65535] to [0.0, 1.0]
    float Rf = static_cast<float>(R) / 65535.0f;
    float Gf = static_cast<float>(G) / 65535.0f;
    float Bf = static_cast<float>(B) / 65535.0f;

    // RGB to YUV conversion (BT.709)
    // Y  =  0.2126 R + 0.7152 G + 0.0722 B
    // U' = -0.1146 R - 0.3854 G + 0.5000 B + 0.5
    // V' =  0.5000 R - 0.4542 G - 0.0458 B + 0.5
    float Yf = 0.2126f * Rf + 0.7152f * Gf + 0.0722f * Bf;
    float Uf = -0.1146f * Rf - 0.3854f * Gf + 0.5000f * Bf + 0.5f;
    float Vf = 0.5000f * Rf - 0.4542f * Gf - 0.0458f * Bf + 0.5f;

    // Clamp YUV values to [0.0, 1.0]
    Yf = fminf(fmaxf(Yf, 0.0f), 1.0f);
    Uf = fminf(fmaxf(Uf, 0.0f), 1.0f);
    Vf = fminf(fmaxf(Vf, 0.0f), 1.0f);

    // Scale Y to [64, 940] and U/V to [64, 960] for P010LE
    // P010LE uses a limited range with Y in [64, 940] and U/V in [64, 960], centered at 512
    uint16_t Y = static_cast<uint16_t>(Yf * 876.0f + 64.0f + 0.5f);
    uint16_t U = static_cast<uint16_t>((Uf - 0.5f) * 448.0f + 512.0f + 0.5f);
    uint16_t V = static_cast<uint16_t>((Vf - 0.5f) * 448.0f + 512.0f + 0.5f);

    // Ensure Y, U, V are within [64, 940] and [64, 960]
    Y = min(max(Y, static_cast<uint16_t>(64)), static_cast<uint16_t>(940));
    U = min(max(U, static_cast<uint16_t>(64)), static_cast<uint16_t>(960));
    V = min(max(V, static_cast<uint16_t>(64)), static_cast<uint16_t>(960));

    // Write Y value without shifting (10-bit data in lower bits)
    int idxY = y * (dstPitchY / sizeof(uint16_t)) + x;
    dstY[idxY] = Y; // P010LE expects 10-bit data in lower bits

    // Handle UV subsampling (4:2:0)
    // Only process even x and y to cover each 2x2 block
    if ((x % 2 == 0) && (y % 2 == 0))
    {
        // Initialize accumulators for U and V
        float U_acc = 0.0f;
        float V_acc = 0.0f;
        int count = 0;

        // Iterate over the 2x2 block
        for (int dy = 0; dy < 2; ++dy)
        {
            for (int dx = 0; dx < 2; ++dx)
            {
                int nx = x + dx;
                int ny = y + dy;

                // Check bounds
                if (nx < width && ny < height)
                {
                    int neighborIdxRGB = ny * (srcPitchRGB / sizeof(uint16_t)) + nx * 3;

                    uint16_t nR = srcRGB[neighborIdxRGB];
                    uint16_t nG = srcRGB[neighborIdxRGB + 1];
                    uint16_t nB = srcRGB[neighborIdxRGB + 2];

                    // Normalize
                    float nRf = static_cast<float>(nR) / 65535.0f;
                    float nGf = static_cast<float>(nG) / 65535.0f;
                    float nBf = static_cast<float>(nB) / 65535.0f;

                    // RGB to YUV conversion (BT.709)
                    float nYf = 0.2126f * nRf + 0.7152f * nGf + 0.0722f * nBf;
                    float nUf = -0.1146f * nRf - 0.3854f * nGf + 0.5000f * nBf + 0.5f;
                    float nVf = 0.5000f * nRf - 0.4542f * nGf - 0.0458f * nBf + 0.5f;

                    // Clamp
                    nYf = fminf(fmaxf(nYf, 0.0f), 1.0f);
                    nUf = fminf(fmaxf(nUf, 0.0f), 1.0f);
                    nVf = fminf(fmaxf(nVf, 0.0f), 1.0f);

                    // Scale
                    uint16_t nU = static_cast<uint16_t>((nUf - 0.5f) * 448.0f + 512.0f + 0.5f);
                    uint16_t nV = static_cast<uint16_t>((nVf - 0.5f) * 448.0f + 512.0f + 0.5f);

                    // Clamp
                    nU = min(max(nU, static_cast<uint16_t>(64)), static_cast<uint16_t>(960));
                    nV = min(max(nV, static_cast<uint16_t>(64)), static_cast<uint16_t>(960));

                    // Accumulate
                    U_acc += static_cast<float>(nU);
                    V_acc += static_cast<float>(nV);
                    count++;
                }
            }
        }

        // Compute the average U and V for the 2x2 block
        uint16_t avgU = (count > 0) ? static_cast<uint16_t>(U_acc / count + 0.5f) : U;
        uint16_t avgV = (count > 0) ? static_cast<uint16_t>(V_acc / count + 0.5f) : V;

        // Index for UV plane (interleaved U and V)
        int uv_x = x / 2;
        int uv_y = y / 2;
        int idxUV = uv_y * (dstPitchUV / sizeof(uint16_t)) + uv_x * 2;

        // Write averaged U and V values without shifting
        dstUV[idxUV]     = avgU; // U
        dstUV[idxUV + 1] = avgV; // V
    }
}

// Launcher function with C linkage
extern "C" void RGBToP010LE_Launcher(const uint16_t* srcRGB, int srcPitchRGB,
                                     uint16_t* dstY, int dstPitchY, uint16_t* dstUV,
                                     int dstPitchUV, int width, int height,
                                     cudaStream_t stream)
{
    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    RGBToP010LEKernel<<<gridSize, blockSize, 0, stream>>>(
        srcRGB, srcPitchRGB, dstY, dstPitchY, dstUV, dstPitchUV, width, height);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    // Optionally synchronize the stream to ensure completion
    // Uncomment the following line if synchronization is needed here
    // cudaStreamSynchronize(stream);
}
