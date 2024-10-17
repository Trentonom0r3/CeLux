// nv12_to_rgb_optimized.cu
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>

extern "C"
{

    // Clamp function for unsigned char using integer arithmetic
    static __device__ unsigned char clamp_uchar_int(int value)
    {
        return static_cast<unsigned char>(value < 0 ? 0 : (value > 255 ? 255 : value));
    }

    // Clamp function for float
    static __device__ float clamp_float(float value)
    {
        return fminf(fmaxf(value, 0.0f), 1.0f);
    }

    // Clamp function for __half
    static __device__ __half clamp_half(__half value)
    {
        return __hmin(__hmax(value, __float2half(0.0f)), __float2half(1.0f));
    }

    // CUDA Kernel for NV12 to RGB conversion with unsigned char output using integer
    // arithmetic
    __global__ void nv12_to_rgb_kernel_uchar(const unsigned char* __restrict__ yPlane,
                                             const unsigned char* __restrict__ uvPlane,
                                             int width, int height, int yStride,
                                             int uvStride,
                                             unsigned char* __restrict__ rgbOutput,
                                             int rgbStride)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x; // Column
        int y = blockIdx.y * blockDim.y + threadIdx.y; // Row

        if (x >= width || y >= height)
            return;

        // Read Y value
        unsigned char Y = yPlane[y * yStride + x];

        // Calculate UV indices (subsampled by 2)
        int uvX = x / 2;
        int uvY = y / 2;
        int uvIndex = uvY * uvStride + 2 * uvX;

        // Read U and V as uchar2
        uchar2 uv = *((const uchar2*)&uvPlane[uvIndex]);
        unsigned char U = uv.x;
        unsigned char V = uv.y;

        // Convert YUV to RGB using integer arithmetic
        int C = static_cast<int>(Y) - 16;
        int D = static_cast<int>(U) - 128;
        int E = static_cast<int>(V) - 128;

        int R = (298 * C + 409 * E + 128) >> 8;
        int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
        int B = (298 * C + 516 * D + 128) >> 8;

        // Clamp the results to [0, 255]
        unsigned char r = clamp_uchar_int(R);
        unsigned char g = clamp_uchar_int(G);
        unsigned char b = clamp_uchar_int(B);

        // Write RGB values
        int rgbIndex = y * rgbStride + x * 3;
        rgbOutput[rgbIndex + 0] = r; // R
        rgbOutput[rgbIndex + 1] = g; // G
        rgbOutput[rgbIndex + 2] = b; // B
    }

    // CUDA Kernel for NV12 to RGB conversion with float output using fused multiply-add
    __global__ void nv12_to_rgb_kernel_float(const unsigned char* __restrict__ yPlane,
                                             const unsigned char* __restrict__ uvPlane,
                                             int width, int height, int yStride,
                                             int uvStride,
                                             float* __restrict__ rgbOutput,
                                             int rgbStride)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x; // Column
        int y = blockIdx.y * blockDim.y + threadIdx.y; // Row

        if (x >= width || y >= height)
            return;

        const float inv255 = 1.0f / 255.0f;
        const float offsetY = 16.0f / 255.0f;

        // Read and normalize Y value
        float Y = yPlane[y * yStride + x] * inv255;

        // Calculate UV indices (subsampled by 2)
        int uvX = x / 2;
        int uvY = y / 2;

        float U = uvPlane[uvY * uvStride + 2 * uvX] * inv255;
        float V = uvPlane[uvY * uvStride + 2 * uvX + 1] * inv255;

        // Convert YUV to RGB (BT.601)
        float C = Y - offsetY;
        float D = U - 0.5f;
        float E = V - 0.5f;

        float R = fmaf(1.596f, E, 1.164f * C);
        float G = fmaf(-0.813f, E, fmaf(-0.392f, D, 1.164f * C));
        float B = fmaf(2.017f, D, 1.164f * C);

        // Clamp the results to [0.0, 1.0]
        R = clamp_float(R);
        G = clamp_float(G);
        B = clamp_float(B);

        // Write RGB values
        int rgbIndex = y * rgbStride + x * 3;
        rgbOutput[rgbIndex + 0] = R; // R
        rgbOutput[rgbIndex + 1] = G; // G
        rgbOutput[rgbIndex + 2] = B; // B
    }

    // CUDA Kernel for NV12 to RGB conversion with __half output using fused
    // multiply-add
    __global__ void nv12_to_rgb_kernel_half(const unsigned char* __restrict__ yPlane,
                                            const unsigned char* __restrict__ uvPlane,
                                            int width, int height, int yStride,
                                            int uvStride,
                                            __half* __restrict__ rgbOutput,
                                            int rgbStride)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x; // Column
        int y = blockIdx.y * blockDim.y + threadIdx.y; // Row

        if (x >= width || y >= height)
            return;

        const __half inv255 = __float2half(1.0f / 255.0f);
        const __half offsetY = __float2half(16.0f / 255.0f);

        // Read and normalize Y value
        __half Y = __hmul(__float2half(yPlane[y * yStride + x]), inv255);

        // Calculate UV indices (subsampled by 2)
        int uvX = x / 2;
        int uvY = y / 2;

        __half U = __hmul(__float2half(uvPlane[uvY * uvStride + 2 * uvX]), inv255);
        __half V = __hmul(__float2half(uvPlane[uvY * uvStride + 2 * uvX + 1]), inv255);

        // Convert YUV to RGB (BT.601)
        __half C = __hsub(Y, offsetY);
        __half D = __hsub(U, __float2half(0.5f));
        __half E = __hsub(V, __float2half(0.5f));

        __half R = __hfma(__float2half(1.596f), E, __hmul(__float2half(1.164f), C));
        __half G =
            __hfma(__float2half(-0.813f), E,
                   __hfma(__float2half(-0.392f), D, __hmul(__float2half(1.164f), C)));
        __half B = __hfma(__float2half(2.017f), D, __hmul(__float2half(1.164f), C));

        // Clamp the results to [0.0, 1.0]
        R = clamp_half(R);
        G = clamp_half(G);
        B = clamp_half(B);

        // Write RGB values
        int rgbIndex = y * rgbStride + x * 3;
        rgbOutput[rgbIndex + 0] = R; // R
        rgbOutput[rgbIndex + 1] = G; // G
        rgbOutput[rgbIndex + 2] = B; // B
    }

    // Host function to launch the unsigned char kernel
    void nv12_to_rgb(const unsigned char* yPlane, const unsigned char* uvPlane,
                     int width, int height, int yStride, int uvStride,
                     unsigned char* rgbOutput, int rgbStride, cudaStream_t stream = 0)
    {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        nv12_to_rgb_kernel_uchar<<<grid, block, 0, stream>>>(
            yPlane, uvPlane, width, height, yStride, uvStride, rgbOutput, rgbStride);

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA kernel launch error (uchar): %s\n",
                    cudaGetErrorString(err));
            throw std::runtime_error("CUDA kernel launch failed (uchar).");
        }
    }

    // Host function to launch the float kernel
    void nv12_to_rgb_float(const unsigned char* yPlane, const unsigned char* uvPlane,
                           int width, int height, int yStride, int uvStride,
                           float* rgbOutput, int rgbStride, cudaStream_t stream = 0)
    {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        nv12_to_rgb_kernel_float<<<grid, block, 0, stream>>>(
            yPlane, uvPlane, width, height, yStride, uvStride, rgbOutput, rgbStride);

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA kernel launch error (float): %s\n",
                    cudaGetErrorString(err));
            throw std::runtime_error("CUDA kernel launch failed (float).");
        }
    }

    // Host function to launch the __half kernel
    void nv12_to_rgb_half(const unsigned char* yPlane, const unsigned char* uvPlane,
                          int width, int height, int yStride, int uvStride,
                          __half* rgbOutput, int rgbStride, cudaStream_t stream = 0)
    {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        nv12_to_rgb_kernel_half<<<grid, block, 0, stream>>>(
            yPlane, uvPlane, width, height, yStride, uvStride, rgbOutput, rgbStride);

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA kernel launch error (__half): %s\n",
                    cudaGetErrorString(err));
            throw std::runtime_error("CUDA kernel launch failed (__half).");
        }
    }

} // extern "C"
// nv12_to_rgb_optimized.cu
