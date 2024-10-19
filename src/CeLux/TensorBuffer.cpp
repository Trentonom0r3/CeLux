#include "TensorBuffer.hpp"
#include <thread>

TensorRingBuffer::TensorRingBuffer(const BufferConfig& config)
    : config_(config), bufferSize_(config.queueSize), readIndex_(0), writeIndex_(0),
      filledSlots_(0), stopped_(false)
{
  
    // Preallocate tensors on the specified device
    buffer_.reserve(bufferSize_);
    for (size_t i = 0; i < bufferSize_; ++i)
    {
        buffer_.emplace_back(torch::empty(config_.shape, torch::TensorOptions()
                                                             .dtype(config_.dtype)
                                                             .device(config_.device)
                                                             .requires_grad(false)));
    }
}

TensorRingBuffer::~TensorRingBuffer()
{
    stop();
  
}

bool TensorRingBuffer::produce(std::function<bool(torch::Tensor&)> fillFunction)
{
    size_t currentWrite = writeIndex_.load(std::memory_order_relaxed);

    // Wait until there is space available or buffer is stopped
    while (filledSlots_.load(std::memory_order_acquire) >= bufferSize_)
    {
        if (stopped_.load(std::memory_order_acquire))
        {
            return false;
        }
        std::this_thread::yield(); // Yield to reduce CPU usage
    }

    if (stopped_.load(std::memory_order_acquire))
    {
        return false;
    }

    torch::Tensor& tensor = buffer_[currentWrite];

    // Perform fill operation
    bool success = fillFunction(tensor);

    if (!success)
    {
        stop();
        return false;
    }

    // Update write index and filled slots atomically
    writeIndex_.store((currentWrite + 1) % bufferSize_, std::memory_order_release);
    filledSlots_.fetch_add(1, std::memory_order_release);

    return true;
}

torch::Tensor TensorRingBuffer::consume()
{
    size_t currentRead = readIndex_.load(std::memory_order_relaxed);

    // Wait until there is data available or buffer is stopped
    while (filledSlots_.load(std::memory_order_acquire) == 0)
    {
        if (stopped_.load(std::memory_order_acquire))
        {
            return torch::Tensor();
        }
        std::this_thread::yield(); // Yield to reduce CPU usage
    }

    torch::Tensor tensor = buffer_[currentRead];

    // Update read index and filled slots atomically
    readIndex_.store((currentRead + 1) % bufferSize_, std::memory_order_release);
    filledSlots_.fetch_sub(1, std::memory_order_release);

    return tensor;
}

void TensorRingBuffer::stop()
{
    bool expected = false;
    if (stopped_.compare_exchange_strong(expected, true, std::memory_order_release))
    {
        // Only notify if this is the first time stop is called
        // Since we removed condition variables, nothing to notify
    }
}

bool TensorRingBuffer::isStopped() const
{
    return stopped_.load(std::memory_order_acquire);
}

size_t TensorRingBuffer::size() const
{
    return filledSlots_.load(std::memory_order_acquire);
}
