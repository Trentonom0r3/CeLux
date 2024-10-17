// Frame.cpp

#include "Frame.hpp"

using namespace celux::error;

namespace celux
{
/**
 * @brief Default constructor that allocates a new AVFrame.
 *
 * @throws FFException if AVFrame allocation fails.
 */
Frame::Frame() : frame(av_frame_alloc())
{
    if (!frame)
    {
        throw CxException("Failed to allocate AVFrame");
    }
}

/**
 * @brief Constructor that takes an existing AVFrame pointer.
 *
 * @param frame Pointer to an existing AVFrame.
 * @throws FFException if the provided AVFrame pointer is null.
 */
Frame::Frame(AVFrame* frame) : frame(frame)
{
    if (!frame)
    {
        throw CxException("Null AVFrame provided");
    }
}

/**
 * @brief Destructor that frees the AVFrame.
 */
Frame::~Frame()
{
    av_frame_free(&frame);
}

/**
 * @brief Copy constructor that clones the AVFrame from another Frame.
 *
 * @param other The Frame object to copy from.
 * @throws FFException if AVFrame cloning fails.
 */
Frame::Frame(const Frame& other) : frame(av_frame_clone(other.frame))
{
    if (!frame)
    {
        throw CxException("Failed to clone AVFrame");
    }
}

/**
 * @brief Copy assignment operator that clones the AVFrame from another Frame.
 *
 * @param other The Frame object to copy from.
 * @return Reference to the assigned Frame object.
 * @throws FFException if AVFrame cloning fails.
 */
Frame& Frame::operator=(const Frame& other)
{
    if (this != &other)
    {
        // Unreference and free the existing frame
        av_frame_unref(frame);
        av_frame_free(&frame);

        // Clone the AVFrame from the other Frame
        frame = av_frame_clone(other.frame);
        if (!frame)
        {
            throw CxException("Failed to clone AVFrame during copy assignment");
        }
    }
    return *this;
}

/**
 * @brief Move constructor that transfers ownership of the AVFrame.
 *
 * @param other The Frame object to move from.
 */
Frame::Frame(Frame&& other) noexcept : frame(other.frame)
{
    other.frame = nullptr;
}

/**
 * @brief Move assignment operator that transfers ownership of the AVFrame.
 *
 * @param other The Frame object to move from.
 * @return Reference to the assigned Frame object.
 */
Frame& Frame::operator=(Frame&& other) noexcept
{
    if (this != &other)
    {
        // Unreference and free the existing frame
        av_frame_unref(frame);
        av_frame_free(&frame);

        // Transfer ownership of the frame pointer
        frame = other.frame;
        other.frame = nullptr;
    }
    return *this;
}

/**
 * @brief Access the underlying AVFrame pointer.
 *
 * @return Pointer to the AVFrame.
 */
AVFrame* Frame::get() const
{
    return frame;
}

/**
 * @brief Get the width of the frame.
 *
 * @return Width in pixels.
 */
int Frame::getWidth() const
{
    return frame->width;
}

/**
 * @brief Get the height of the frame.
 *
 * @return Height in pixels.
 */
int Frame::getHeight() const
{
    return frame->height;
}

/**
 * @brief Get the pixel format of the frame.
 *
 * @return AVPixelFormat enumeration value.
 */
AVPixelFormat Frame::getPixelFormat() const
{
    return static_cast<AVPixelFormat>(frame->format);
}

std::string Frame::getPixelFormatString() const
{
    return av_get_pix_fmt_name(getPixelFormat());
}

/**
 * @brief Get the data pointer for a specific plane.
 *
 * @param plane Plane index (0 for Y, 1 for U, 2 for V in YUV formats).
 * @return Pointer to the data of the specified plane.
 * @throws FFException if the plane index is out of range.
 */
uint8_t* Frame::getData(int plane) const
{
    if (plane < 0 || plane >= AV_NUM_DATA_POINTERS)
    {
        throw CxException("Invalid plane index: " + std::to_string(plane));
    }
    return frame->data[plane];
}

/**
 * @brief Get the line size for a specific plane.
 *
 * @param plane Plane index.
 * @return Line size in bytes.
 * @throws FFException if the plane index is out of range.
 */
int Frame::getLineSize(int plane) const
{
    if (plane < 0 || plane >= AV_NUM_DATA_POINTERS)
    {
        throw CxException("Invalid plane index: " + std::to_string(plane));
    }
    return frame->linesize[plane];
}

/**
 * @brief Get the presentation timestamp of the frame.
 *
 * @return Presentation timestamp (PTS).
 */
int64_t Frame::getPts() const
{
    return frame->pts;
}

/**
 * @brief Set the presentation timestamp of the frame.
 *
 * @param pts New presentation timestamp.
 */
void Frame::setPts(int64_t pts)
{
    frame->pts = pts;
}

/**
 * @brief Check if the Frame holds a valid AVFrame.
 *
 * @return True if the AVFrame is valid, False otherwise.
 */
Frame::operator bool() const
{
    return frame != nullptr;
}

/**
 * @brief Allocate buffer for the AVFrame with the specified alignment.
 *
 * @param align Alignment in bytes (e.g., 32 for SIMD optimizations).
 * @throws FFException if buffer allocation fails.
 */
void Frame::allocateBuffer(int align)
{
    if (av_frame_get_buffer(frame, align) < 0)
    {
        throw CxException("Failed to allocate buffer for AVFrame with alignment " +
                          std::to_string(align));
    }
}

/**
 * @brief Copy frame data from another Frame.
 *
 * @param other The source Frame to copy data from.
 * @throws FFException if frame copying fails.
 */
void Frame::copyFrom(const Frame& other)
{
    if (av_frame_copy(frame, other.frame) < 0)
    {
        throw CxException("Failed to copy data from source AVFrame");
    }
    if (av_frame_copy_props(frame, other.frame) < 0)
    {
        throw CxException("Failed to copy properties from source AVFrame");
    }
}

/**
 * @brief Fill the frame with raw data.
 *
 * @param data Pointer to the raw data buffer.
 * @param size Size of the data buffer in bytes.
 * @param plane Plane index to fill.
 * @throws FFException if data copying fails or plane index is invalid.
 */
void Frame::fillData(uint8_t* data, int size, int plane)
{
    if (plane < 0 || plane >= AV_NUM_DATA_POINTERS)
    {
        throw CxException("Invalid plane index: " + std::to_string(plane));
    }

    int planeHeight = (plane == 0) ? getHeight() : (getHeight() + 1) / 2;
    int maxSize = frame->linesize[plane] * planeHeight;

    if (size > maxSize)
    {
        throw CxException("Data size exceeds buffer capacity for plane " +
                          std::to_string(plane));
    }

    memcpy(frame->data[plane], data, size);
}

/**
 * @brief Overload the << operator to print Frame information.
 *
 * @param os Output stream.
 * @param frame The Frame object to print.
 * @return Reference to the output stream.
 */
std::ostream& operator<<(std::ostream& os, const Frame& frame)
{
    os << "Frame(width=" << frame.getWidth() << ", height=" << frame.getHeight()
       << ", format=" << av_get_pix_fmt_name(frame.getPixelFormat())
       << ", pts=" << frame.getPts() << ")";
    return os;
}
} // namespace celux
