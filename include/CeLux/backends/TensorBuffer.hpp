#pragma once
#ifndef TENSORBUFFER_HPP
#define TENSORBUFFER_HPP

#include <atomic>
#include <functional>
#include <torch/torch.h>

struct BufferConfig
{
    torch::IntArrayRef shape;
    torch::Dtype dtype;
    torch::Device device;
    size_t queueSize;

    BufferConfig(const torch::IntArrayRef& shape, torch::Dtype dtype,
                 const torch::Device& device, size_t queueSize)
        : shape(shape), dtype(dtype), device(device), queueSize(queueSize)
    {
    }

    BufferConfig() : shape(), dtype(torch::kUInt8), device(torch::kCPU), queueSize(1)
    {
    }
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
