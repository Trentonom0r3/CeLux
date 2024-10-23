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
    CELUX_DEBUG("Frame constructor called: Allocating new AVFrame");
    if (!frame)
    {
        CELUX_ERROR("Failed to allocate AVFrame in default constructor");
        throw CxException("Failed to allocate AVFrame");
    }
    CELUX_INFO("AVFrame allocated successfully in default constructor");
}

/**
 * @brief Constructor that takes an existing AVFrame pointer.
 *
 * @param frame Pointer to an existing AVFrame.
 * @throws FFException if the provided AVFrame pointer is null.
 */
Frame::Frame(AVFrame* frame) : frame(frame)
{
    CELUX_DEBUG("Frame constructor called with existing AVFrame pointer");
    if (!frame)
    {
        CELUX_ERROR("Null AVFrame provided to constructor");
        throw CxException("Null AVFrame provided");
    }
    CELUX_INFO("Frame initialized with existing AVFrame pointer");
}

/**
 * @brief Destructor that frees the AVFrame.
 */
Frame::~Frame()
{
    CELUX_DEBUG("Frame destructor called: Freeing AVFrame");
    av_frame_free(&frame);
    CELUX_INFO("AVFrame freed successfully in destructor");
}

/**
 * @brief Copy constructor that clones the AVFrame from another Frame.
 *
 * @param other The Frame object to copy from.
 * @throws FFException if AVFrame cloning fails.
 */
Frame::Frame(const Frame& other) : frame(av_frame_clone(other.frame))
{
    CELUX_DEBUG("Frame copy constructor called");
    if (!frame)
    {
        CELUX_ERROR("Failed to clone AVFrame in copy constructor");
        throw CxException("Failed to clone AVFrame");
    }
    CELUX_INFO("AVFrame cloned successfully in copy constructor");
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
    CELUX_DEBUG("Frame copy assignment operator called");
    if (this != &other)
    {
        // Unreference and free the existing frame
        CELUX_DEBUG("Unreferencing and freeing existing AVFrame in copy assignment");
        av_frame_unref(frame);
        av_frame_free(&frame);

        // Clone the AVFrame from the other Frame
        CELUX_DEBUG("Cloning AVFrame from source Frame");
        frame = av_frame_clone(other.frame);
        if (!frame)
        {
            CELUX_ERROR("Failed to clone AVFrame during copy assignment");
            throw CxException("Failed to clone AVFrame during copy assignment");
        }
        CELUX_INFO("AVFrame cloned successfully in copy assignment");
    }
    else
    {
        CELUX_WARN("Self-assignment detected in copy assignment operator");
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
    CELUX_DEBUG("Frame move constructor called: Transferring AVFrame ownership");
    other.frame = nullptr;
    CELUX_INFO("AVFrame ownership transferred successfully in move constructor");
}

/**
 * @brief Move assignment operator that transfers ownership of the AVFrame.
 *
 * @param other The Frame object to move from.
 * @return Reference to the assigned Frame object.
 */
Frame& Frame::operator=(Frame&& other) noexcept
{
    CELUX_DEBUG("Frame move assignment operator called");
    if (this != &other)
    {
        // Unreference and free the existing frame
        CELUX_DEBUG("Unreferencing and freeing existing AVFrame in move assignment");
        av_frame_unref(frame);
        av_frame_free(&frame);

        // Transfer ownership of the frame pointer
        frame = other.frame;
        other.frame = nullptr;
        CELUX_INFO("AVFrame ownership transferred successfully in move assignment");
    }
    else
    {
        CELUX_WARN("Self-assignment detected in move assignment operator");
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
    CELUX_TRACE("Frame::get() called: Returning AVFrame pointer");
    return frame;
}

/**
 * @brief Get the width of the frame.
 *
 * @return Width in pixels.
 */
int Frame::getWidth() const
{
    CELUX_TRACE("Frame::getWidth() called: width = {}", frame->width);
    return frame->width;
}

/**
 * @brief Get the height of the frame.
 *
 * @return Height in pixels.
 */
int Frame::getHeight() const
{
    CELUX_TRACE("Frame::getHeight() called: height = {}", frame->height);
    return frame->height;
}

/**
 * @brief Get the pixel format of the frame.
 *
 * @return AVPixelFormat enumeration value.
 */
AVPixelFormat Frame::getPixelFormat() const
{
    CELUX_TRACE("Frame::getPixelFormat() called: format = {}",
                av_get_pix_fmt_name(static_cast<AVPixelFormat>(frame->format)));
    return static_cast<AVPixelFormat>(frame->format);
}

std::string Frame::getPixelFormatString() const
{
    std::string pix_fmt_str = av_get_pix_fmt_name(getPixelFormat())
                                  ? av_get_pix_fmt_name(getPixelFormat())
                                  : "Unknown";
    CELUX_TRACE("Frame::getPixelFormatString() called: format = {}", pix_fmt_str);
    return pix_fmt_str;
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
    CELUX_TRACE("Frame::getData() called with plane = {}", plane);
    if (plane < 0 || plane >= AV_NUM_DATA_POINTERS)
    {
        CELUX_ERROR("Invalid plane index in getData(): {}", plane);
        throw CxException("Invalid plane index: " + std::to_string(plane));
    }
    CELUX_DEBUG("Returning data pointer for plane {}: {}", plane,
                static_cast<void*>(frame->data[plane]));
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
    CELUX_TRACE("Frame::getLineSize() called with plane = {}", plane);
    if (plane < 0 || plane >= AV_NUM_DATA_POINTERS)
    {
        CELUX_ERROR("Invalid plane index in getLineSize(): {}", plane);
        throw CxException("Invalid plane index: " + std::to_string(plane));
    }
    CELUX_DEBUG("Returning line size for plane {}: {}", plane, frame->linesize[plane]);
    return frame->linesize[plane];
}

/**
 * @brief Get the presentation timestamp of the frame.
 *
 * @return Presentation timestamp (PTS).
 */
int64_t Frame::getPts() const
{
    CELUX_TRACE("Frame::getPts() called: pts = {}", frame->pts);
    return frame->pts;
}

/**
 * @brief Set the presentation timestamp of the frame.
 *
 * @param pts New presentation timestamp.
 */
void Frame::setPts(int64_t pts)
{
    CELUX_TRACE("Frame::setPts() called: setting pts from {} to {}", frame->pts, pts);
    frame->pts = pts;
    CELUX_DEBUG("Frame PTS set to {}", pts);
}

/**
 * @brief Check if the Frame holds a valid AVFrame.
 *
 * @return True if the AVFrame is valid, False otherwise.
 */
Frame::operator bool() const
{
    bool isValid = frame != nullptr;
    CELUX_TRACE("Frame::operator bool() called: isValid = {}", isValid);
    return isValid;
}

/**
 * @brief Allocate buffer for the AVFrame with the specified alignment.
 *
 * @param align Alignment in bytes (e.g., 32 for SIMD optimizations).
 * @throws FFException if buffer allocation fails.
 */
void Frame::allocateBuffer(int align)
{
    CELUX_INFO("Frame::allocateBuffer() called with alignment = {}", align);
    if (av_frame_get_buffer(frame, align) < 0)
    {
        CELUX_ERROR("Failed to allocate buffer for AVFrame with alignment {}", align);
        throw CxException("Failed to allocate buffer for AVFrame with alignment " +
                          std::to_string(align));
    }
    CELUX_DEBUG("Buffer allocated for AVFrame with alignment {}", align);
}

/**
 * @brief Copy frame data from another Frame.
 *
 * @param other The source Frame to copy data from.
 * @throws FFException if frame copying fails.
 */
void Frame::copyFrom(const Frame& other)
{
    CELUX_INFO("Frame::copyFrom() called: Copying data from another Frame");
    if (av_frame_copy(frame, other.frame) < 0)
    {
        CELUX_ERROR("Failed to copy data from source AVFrame");
        throw CxException("Failed to copy data from source AVFrame");
    }
    CELUX_DEBUG("Frame data copied successfully from source AVFrame");

    if (av_frame_copy_props(frame, other.frame) < 0)
    {
        CELUX_ERROR("Failed to copy properties from source AVFrame");
        throw CxException("Failed to copy properties from source AVFrame");
    }
    CELUX_DEBUG("Frame properties copied successfully from source AVFrame");
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
    CELUX_TRACE("Frame::fillData() called with plane = {}, size = {}", plane, size);
    if (plane < 0 || plane >= AV_NUM_DATA_POINTERS)
    {
        CELUX_ERROR("Invalid plane index in fillData(): {}", plane);
        throw CxException("Invalid plane index: " + std::to_string(plane));
    }

    int planeHeight = (plane == 0) ? getHeight() : (getHeight() + 1) / 2;
    int maxSize = frame->linesize[plane] * planeHeight;
    CELUX_DEBUG("Plane {}: planeHeight = {}, maxSize = {}", plane, planeHeight,
                maxSize);

    if (size > maxSize)
    {
        CELUX_ERROR("Data size {} exceeds buffer capacity {} for plane {}", size,
                    maxSize, plane);
        throw CxException("Data size exceeds buffer capacity for plane " +
                          std::to_string(plane));
    }

    memcpy(frame->data[plane], data, size);
    CELUX_INFO("Data filled into plane {} successfully, size = {}", plane, size);
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
    CELUX_TRACE("operator<< called to output Frame information");
    os << "Frame(width=" << frame.getWidth() << ", height=" << frame.getHeight()
       << ", format=" << frame.getPixelFormatString() << ", pts=" << frame.getPts()
       << ")";
    return os;
}

} // namespace celux
