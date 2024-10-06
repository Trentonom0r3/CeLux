// nv12_to_rgb.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>

extern "C" {

    // Clamp function to ensure RGB values are within [0, 255]
    __device__ unsigned char clamp(float value) {
        return static_cast<unsigned char>(value < 0 ? 0 : (value > 255 ? 255 : value));
    }

    // CUDA Kernel for NV12 to RGB conversion
    __global__ void nv12_to_rgb_kernel(
        const unsigned char* __restrict__ yPlane,
        const unsigned char* __restrict__ uvPlane,
        int width,
        int height,
        int yStride,
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

        unsigned char U = uvPlane[uvY * uvStride + 2 * uvX];
        unsigned char V = uvPlane[uvY * uvStride + 2 * uvX + 1];

        // Convert YUV to RGB (BT.601)
        float C = Y - 16.0f;
        float D = U - 128.0f;
        float E = V - 128.0f;

        float R = 1.164f * C + 1.596f * E;
        float G = 1.164f * C - 0.392f * D - 0.813f * E;
        float B = 1.164f * C + 2.017f * D;

        // Clamp the results to [0, 255]
        unsigned char r = clamp(R);
        unsigned char g = clamp(G);
        unsigned char b = clamp(B);

        // Write RGB values
        int rgbIndex = y * rgbStride + x * 3;
        rgbOutput[rgbIndex + 0] = r; // R
        rgbOutput[rgbIndex + 1] = g; // G
        rgbOutput[rgbIndex + 2] = b; // B
    }

    // Host function to launch the kernel
    void nv12_to_rgb(
        const unsigned char* yPlane,
        const unsigned char* uvPlane,
        int width,
        int height,
        int yStride,
        int uvStride,
        unsigned char* rgbOutput,
        int rgbStride,
        cudaStream_t stream = 0)
    {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        nv12_to_rgb_kernel << <grid, block, 0, stream >> > (
			yPlane,
			uvPlane,
			width,
			height,
			yStride,
			uvStride,
			rgbOutput,
			rgbStride
		);

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
            throw std::runtime_error("CUDA kernel launch failed.");
        }
    }
}  // extern "C"
