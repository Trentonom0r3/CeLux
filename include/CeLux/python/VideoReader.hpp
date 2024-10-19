// VideoReader.hpp

#ifndef VIDEOREADER_HPP
#define VIDEOREADER_HPP

#include "Factory.hpp"
#include <TensorBuffer.hpp>

#include <pybind11/pybind11.h>
namespace py = pybind11;

// Enum for copy types
enum class CopyType
{
    HOST,
    DEVICE
};

class VideoReader
{
  public:

    VideoReader(const std::string& filePath, const std::string& device = "cuda",
                const std::string& dtype = "uint8", const int bufferSize = 10,
                std::optional<torch::Stream> stream = std::nullopt);


    /**
     * @brief Destructor for VideoReader.
     */
    ~VideoReader();

    /**
     * @brief Read a frame from the video.
     *
     * Depending on the configuration, returns either a torch::Tensor or a
     * py::array<uint8_t>. Shape is always HWC. If batch size is specified in Reader
     * config, output shape will be BHWC for Tensors.
     *
     * @return  torch::Tensor (torch::Tensor or py::array<uint8_t>)
     */
    torch::Tensor readFrame();

    /**
     * @brief Seek to a specific timestamp in the video.
     *
     * @param timestamp The timestamp to seek to (in seconds).
     * @return true if seek was successful, false otherwise.
     */
    bool seek(double timestamp);

    /**
     * @brief Get a list of supported codecs.
     *
     * @return std::vector<std::string> List of supported codec names.
     */
    std::vector<std::string> supportedCodecs();

    /**
     * @brief Get the properties of the video.
     *
     * @return py::dict Dictionary containing video properties.
     */
    py::dict getProperties() const;

    /**
     * @brief Reset the video reader to the beginning.
     */
    void reset();

    /**
     * @brief Iterator initialization for Python.
     *
     * @return VideoReader& Reference to the VideoReader object.
     */
    VideoReader& iter();

    /**
     * @brief Get the next frame in iteration.
     *
     * @return  torch::Tensor Next frame as torch::Tensor or py::array<uint8_t>.
     */
     torch::Tensor next();

    /**
     * @brief Enter method for Python context manager.
     */
    void enter();

    /**
     * @brief Exit method for Python context manager.
     *
     * @param exc_type Exception type (if any).
     * @param exc_value Exception value (if any).
     * @param traceback Traceback object (if any).
     */
    void exit(const py::object& exc_type, const py::object& exc_value,
              const py::object& traceback);
    int length() const;
    void setRange(int start, int end);

    void sync();

  private:
    void bufferFrames();
    std::unique_ptr<TensorRingBuffer> tensorBuffer_;
    std::thread bufferThread_;
    std::atomic<bool> stopBuffering_;
    torch::Dtype torchDataType_;
    bool seekToFrame(int frame_number);

    /**
     * @brief Close the video reader and release resources.
     */
    void close();

    // Member variables
    std::unique_ptr<celux::Decoder> decoder;
    celux::Decoder::VideoProperties properties;
    std::string device;

    torch::Device torchDevice;

    std::unique_ptr<celux::conversion::IConverter> convert;

    // Buffers
    torch::Tensor RGBTensor; // For RGB conversion (GPU)
    torch::Tensor cpuTensor; // For CPU conversion (CPU)
    celux::Frame frame;      // Decoded frame
    int start_frame = 0;
    int end_frame = -1; // -1 indicates no limit

    // Iterator state
    int currentIndex;
};

#endif // VIDEOREADER_HPP
