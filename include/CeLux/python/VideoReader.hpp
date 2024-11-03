// VideoReader.hpp

#ifndef VIDEOREADER_HPP
#define VIDEOREADER_HPP

#include "Decoder.hpp" // Ensure this includes the Filter class
#include "Factory.hpp"
#include <memory>     // For std::unique_ptr
#include <pybind11/pybind11.h>
#include <string> // For std::string
#include <vector> // For std::vector


namespace py = pybind11;

class VideoReader
{
  public:
    /**
     * @brief Constructs a VideoReader for a given input file.
     *
     * @param filePath Path to the input video file.
     * @param numThreads Number of threads to use for decoding.
     * @param device Processing device ("cpu" or "cuda").
     */
    VideoReader(const std::string& filePath,
                int numThreads = static_cast<int>(std::thread::hardware_concurrency() /
                                                  2),
                const std::string& device = "cuda",
                std::vector<std::shared_ptr<FilterBase>> filter = {});

    /**
     * @brief Destructor for VideoReader.
     */
    ~VideoReader();

    /**
     * @brief Overloads the [] operator to access video properties by key.
     *
     * @param key The property key to access.
     * @return py::object The value associated with the key.
     */
    py::object operator[](const std::string& key) const;

    /**
     * @brief Read a frame from the video.
     *
     * Depending on the configuration, returns either a torch::Tensor or a
     * py::array<uint8_t>. Shape is always HWC. If batch size is specified in Reader
     * config, output shape will be BHWC for Tensors.
     *
     * @return torch::Tensor The next frame as torch::Tensor.
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
     * @return torch::Tensor Next frame as torch::Tensor.
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

    /**
     * @brief Get the total number of frames.
     *
     * @return int Total frame count.
     */
    int length() const;

    /**
     * @brief Set the range of frames to read.
     *
     * @param start Starting frame index.
     * @param end Ending frame index (-1 for no limit).
     */
    void setRange(int start, int end);

    /**
     * @brief Add a filter to the decoder's filter pipeline.
     *
     * @param filterName Name of the filter (e.g., "scale").
     * @param filterOptions Options for the filter (e.g., "1280:720").
     */
    void addFilter(const std::string& filterName, const std::string& filterOptions);

    /**
     * @brief Initialize the decoder after adding all desired filters.
     *
     * This separates filter addition from decoder initialization, allowing
     * users to configure filters before starting the decoding process.
     *
     * @return true if initialization is successful.
     * @return false otherwise.
     */
    bool initialize();

  private:
    bool seekToFrame(int frame_number);
    torch::ScalarType findTypeFromBitDepth();

    /**
     * @brief Close the video reader and release resources.
     */
    void close();

    // Member variables
    std::unique_ptr<celux::Decoder> decoder;
    celux::Decoder::VideoProperties properties;

    torch::Tensor tensor;

    int start_frame = 0;
    int end_frame = -1; // -1 indicates no limit

    // Iterator state
    int currentIndex;

    // List of filters to be added before initialization
    std::vector<std::shared_ptr<FilterBase>> filters_;
};

#endif // VIDEOREADER_HPP
