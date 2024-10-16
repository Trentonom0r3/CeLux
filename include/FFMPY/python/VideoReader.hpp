// VideoReader.hpp

#ifndef VIDEOREADER_HPP
#define VIDEOREADER_HPP

#include "Factory.hpp"
#include <torch/extension.h>
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
    /**
     * @brief Constructs a VideoReader object.
     *
     * @param filePath Path to the video file.
     * @param useHardware Flag to indicate whether to use hardware acceleration.
     * @param hwType Type of hardware acceleration (e.g., "cuda").
     * @param config Configuration for frame processing.
     */
    VideoReader(const std::string& filePath, const std::string& device = "cuda",
                const std::string& dtype = "uint8");

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
    /**
     * @brief Copy data from source to destination based on the copy type.
     *
     * @param src Source pointer.
     * @param dst Destination pointer.
     * @param size Number of bytes to copy.
     * @param type Type of copy (HOST or DEVICE).
     */
    void copyTo(void* src, void* dst, size_t size, CopyType type);
    bool seekToFrame(int frame_number);

    /**
     * @brief Close the video reader and release resources.
     */
    void close();

    // Member variables
    std::unique_ptr<ffmpy::Decoder> decoder;
    ffmpy::Decoder::VideoProperties properties;
    std::string device;

    torch::Device torchDevice;

    std::unique_ptr<ffmpy::conversion::IConverter> convert;

    // Buffers
    torch::Tensor RGBTensor; // For RGB conversion (GPU)
    torch::Tensor cpuTensor; // For CPU conversion (CPU)
    ffmpy::Frame frame;      // Decoded frame
    int start_frame = 0;
    int end_frame = -1; // -1 indicates no limit

    // Iterator state
    int currentIndex;
};

#endif // VIDEOREADER_HPP
