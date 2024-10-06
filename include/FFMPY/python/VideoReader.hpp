// VideoReader.hpp

#ifndef VIDEOREADER_HPP
#define VIDEOREADER_HPP

#include "Decoder.hpp"
#include "NV12ToRGB.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <NV12ToRGB.hpp>


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
    VideoReader(const std::string& filePath, bool useHardware = true,
                const std::string& hwType = "cuda", bool as_numpy = false);

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
     * @return py::object (torch::Tensor or py::array<uint8_t>)
     */
    py::object readFrame();

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
     * @return py::object Next frame as torch::Tensor or py::array<uint8_t>.
     */
    py::object next();

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
  private:
    /**
     * @brief Copy data from source to destination based on the copy type.
     *
     * @param src Source pointer.
     * @param dst Destination pointer.
     * @param size Number of bytes to copy.
     * @param type Type of copy (HOST or DEVICE).
     */
    void copyTo(uint8_t* src, uint8_t* dst, size_t size, CopyType type);

    /**
     * @brief Close the video reader and release resources.
     */
    void close();

    // Member variables
    std::unique_ptr<ffmpy::Decoder> decoder;
    ffmpy::Decoder::VideoProperties properties;
    bool as_numpy;
    ffmpy::conversion::NV12ToRGB<uint8_t> convert;

    // Buffers
    torch::Tensor rgb_tensor;      // For RGB conversion (GPU)
    torch::Tensor tensorBuffer;    // For Tensor Output (CPU or GPU)
    py::array_t<uint8_t> npBuffer; // For NumPy Output
    ffmpy::Frame frame;            // Decoded frame

    // Iterator state
    int currentIndex;
};

#endif // VIDEOREADER_HPP
