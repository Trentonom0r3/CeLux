#pragma once
#ifndef TENSORBUFFER_HPP
#define TENSORBUFFER_HPP

#include <atomic>
#include <functional>
#include <torch/torch.h>

struct BufferConfig
{
    int height;
    int width;
    int channels = 3;
    torch::Dtype dtype;
    torch::Device device;
    size_t queueSize;

    BufferConfig(int height, int width, int channels, torch::Dtype dtype,
                 const torch::Device& device, size_t queueSize)
        : height(height), width(width), channels(channels), dtype(dtype),
        device(device), queueSize(queueSize)
    {
    
    }

    BufferConfig() : BufferConfig(0, 0, 3, torch::kUInt8, torch::kCPU, 1)
	{}
};

class TensorRingBuffer
{
  public:
    explicit TensorRingBuffer(const BufferConfig& config);
    ~TensorRingBuffer();

    // Fill the next available tensor (called by producer)
    bool produce(std::function<bool(torch::Tensor&)> fillFunction);

    // Get the next tensor (called by consumer)
    torch::Tensor consume();

    // Stop the buffer
    void stop();

    // Check if the buffer is stopped
    bool isStopped() const;

    size_t size() const;

  private:
    BufferConfig config_;

    std::vector<torch::Tensor> buffer_;
    size_t bufferSize_;

    std::atomic<size_t> readIndex_;
    std::atomic<size_t> writeIndex_;
    std::atomic<size_t> filledSlots_;

    std::atomic<bool> stopped_;


};

#endif // TENSORBUFFER_HPP
