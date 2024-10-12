// rgb_to_nv12_optimized.cu
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>

extern "C"
{
    // Clamp function for unsigned char
    __device__ unsigned char clamp_uchar(int value)
    {
        return static_cast<unsigned char>(value < 0 ? 0 : (value > 255 ? 255 : value));
    }

    // CUDA Kernel for RGB to NV12 conversion with unsigned char input
    __global__ void rgb_to_nv12_kernel_uchar(const unsigned char* __restrict__ rgbInput,
                                             int width, int height, int rgbStride,
                                             unsigned char* __restrict__ yPlane,
                                             unsigned char* __restrict__ uvPlane,
                                             int yStride, int uvStride)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x; // Column
        int y = blockIdx.y * blockDim.y + threadIdx.y; // Row

        if (x >= width || y >= height)
            return;

        // Read RGB values
        int rgbIndex = y * rgbStride + x * 3;
        unsigned char R = rgbInput[rgbIndex + 0];
        unsigned char G = rgbInput[rgbIndex + 1];
        unsigned char B = rgbInput[rgbIndex + 2];

        // Convert RGB to YUV (BT.601)
        int Y = (77 * R + 150 * G + 29 * B) >> 8;
        int U = ((-43 * R - 85 * G + 128 * B) >> 8) + 128;
        int V = ((128 * R - 107 * G - 21 * B) >> 8) + 128;

        // Write Y value
        yPlane[y * yStride + x] = clamp_uchar(Y);

        // Write UV values (subsampled by 2)
        if (x % 2 == 0 && y % 2 == 0)
        {
            int uvIndex = (y / 2) * uvStride + (x / 2) * 2;
            uvPlane[uvIndex + 0] = clamp_uchar(U);
            uvPlane[uvIndex + 1] = clamp_uchar(V);
        }
    }

    // CUDA Kernel for RGB to NV12 conversion with float input
    __global__ void rgb_to_nv12_kernel_float(const float* __restrict__ rgbInput,
                                             int width, int height, int rgbStride,
                                             unsigned char* __restrict__ yPlane,
                                             unsigned char* __restrict__ uvPlane,
                                             int yStride, int uvStride)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x; // Column
        int y = blockIdx.y * blockDim.y + threadIdx.y; // Row

        if (x >= width || y >= height)
            return;

        // Read and denormalize RGB values
        int rgbIndex = y * rgbStride + x * 3;
        float R = rgbInput[rgbIndex + 0] * 255.0f;
        float G = rgbInput[rgbIndex + 1] * 255.0f;
        float B = rgbInput[rgbIndex + 2] * 255.0f;

        // Convert RGB to YUV (BT.601)
        float Y = 0.299f * R + 0.587f * G + 0.114f * B;
        float U = -0.169f * R - 0.331f * G + 0.500f * B + 128.0f;
        float V = 0.500f * R - 0.419f * G - 0.081f * B + 128.0f;

        // Write Y value
        yPlane[y * yStride + x] = clamp_uchar(static_cast<int>(Y));

        // Write UV values (subsampled by 2)
        if (x % 2 == 0 && y % 2 == 0)
        {
            int uvIndex = (y / 2) * uvStride + (x / 2) * 2;
            uvPlane[uvIndex + 0] = clamp_uchar(static_cast<int>(U));
            uvPlane[uvIndex + 1] = clamp_uchar(static_cast<int>(V));
        }
    }

    // CUDA Kernel for RGB to NV12 conversion with __half input
    __global__ void rgb_to_nv12_kernel_half(const __half* __restrict__ rgbInput,
                                            int width, int height, int rgbStride,
                                            unsigned char* __restrict__ yPlane,
                                            unsigned char* __restrict__ uvPlane,
                                            int yStride, int uvStride)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x; // Column
        int y = blockIdx.y * blockDim.y + threadIdx.y; // Row

        if (x >= width || y >= height)
            return;

        // Read and denormalize RGB values
        int rgbIndex = y * rgbStride + x * 3;
        float R = __half2float(rgbInput[rgbIndex + 0]) * 255.0f;
        float G = __half2float(rgbInput[rgbIndex + 1]) * 255.0f;
        float B = __half2float(rgbInput[rgbIndex + 2]) * 255.0f;

        // Convert RGB to YUV (BT.601)
        float Y = 0.299f * R + 0.587f * G + 0.114f * B;
        float U = -0.169f * R - 0.331f * G + 0.500f * B + 128.0f;
        float V = 0.500f * R - 0.419f * G - 0.081f * B + 128.0f;

        // Write Y value
        yPlane[y * yStride + x] = clamp_uchar(static_cast<int>(Y));

        // Write UV values (subsampled by 2)
        if (x % 2 == 0 && y % 2 == 0)
        {
            int uvIndex = (y / 2) * uvStride + (x / 2) * 2;
            uvPlane[uvIndex + 0] = clamp_uchar(static_cast<int>(U));
            uvPlane[uvIndex + 1] = clamp_uchar(static_cast<int>(V));
        }
    }

    // Host function to launch the unsigned char kernel
    void rgb_to_nv12(const unsigned char* rgbInput, int width, int height,
                     int rgbStride, unsigned char* yPlane, unsigned char* uvPlane,
                     int yStride, int uvStride, cudaStream_t stream = 0)
    {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        rgb_to_nv12_kernel_uchar<<<grid, block, 0, stream>>>(
            rgbInput, width, height, rgbStride, yPlane, uvPlane, yStride, uvStride);

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA kernel launch error (uchar): %s\n",
                    cudaGetErrorString(err));
            throw std::runtime_error("CUDA kernel launch failed (uchar).\n");
        }
    }

    // Host function to launch the float kernel
    void rgb_to_nv12_float(const float* rgbInput, int width, int height, int rgbStride,
                           unsigned char* yPlane, unsigned char* uvPlane, int yStride,
                           int uvStride, cudaStream_t stream = 0)
    {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        rgb_to_nv12_kernel_float<<<grid, block, 0, stream>>>(
            rgbInput, width, height, rgbStride, yPlane, uvPlane, yStride, uvStride);

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA kernel launch error (float): %s\n",
                    cudaGetErrorString(err));
            throw std::runtime_error("CUDA kernel launch failed (float).\n");
        }
    }

    // Host function to launch the __half kernel
    void rgb_to_nv12_half(const __half* rgbInput, int width, int height, int rgbStride,
                          unsigned char* yPlane, unsigned char* uvPlane, int yStride,
                          int uvStride, cudaStream_t stream = 0)
    {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        rgb_to_nv12_kernel_half<<<grid, block, 0, stream>>>(
            rgbInput, width, height, rgbStride, yPlane, uvPlane, yStride, uvStride);

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA kernel launch error (__half): %s\n",
                    cudaGetErrorString(err));
            throw std::runtime_error("CUDA kernel launch failed (__half).\n");
        }
    }

} // extern "C"
  // rgb_to_nv12_optimized.cu