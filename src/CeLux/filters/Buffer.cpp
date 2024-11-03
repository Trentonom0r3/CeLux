#include "Buffer.hpp"
#include <sstream>

Buffer::Buffer(int height, std::pair<int, int> video_size, const std::string& pix_fmt, std::pair<int, int> pixel_aspect, std::pair<int, int> frame_rate, int colorspace, int range) {
    // Initialize member variables from parameters
    this->height_ = height;
    this->video_size_ = video_size;
    this->pix_fmt_ = pix_fmt;
    this->pixel_aspect_ = pixel_aspect;
    this->frame_rate_ = frame_rate;
    this->colorspace_ = colorspace;
    this->range_ = range;
}

Buffer::~Buffer() {
    // Destructor implementation (if needed)
}

void Buffer::setHeight(int value) {
    height_ = value;
}

int Buffer::getHeight() const {
    return height_;
}

void Buffer::setVideo_size(const std::pair<int, int>& value) {
    video_size_ = value;
}

std::pair<int, int> Buffer::getVideo_size() const {
    return video_size_;
}

void Buffer::setPix_fmt(const std::string& value) {
    pix_fmt_ = value;
}

std::string Buffer::getPix_fmt() const {
    return pix_fmt_;
}

void Buffer::setPixel_aspect(const std::pair<int, int>& value) {
    pixel_aspect_ = value;
}

std::pair<int, int> Buffer::getPixel_aspect() const {
    return pixel_aspect_;
}

void Buffer::setFrame_rate(const std::pair<int, int>& value) {
    frame_rate_ = value;
}

std::pair<int, int> Buffer::getFrame_rate() const {
    return frame_rate_;
}

void Buffer::setColorspace(int value) {
    colorspace_ = value;
}

int Buffer::getColorspace() const {
    return colorspace_;
}

void Buffer::setRange(int value) {
    range_ = value;
}

int Buffer::getRange() const {
    return range_;
}

std::string Buffer::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "buffer";

    bool first = true;

    if (height_ != 0) {
        desc << (first ? "=" : ":") << "height=" << height_;
        first = false;
    }
    if (video_size_.first != 0 || video_size_.second != 1) {
        desc << (first ? "=" : ":") << "video_size=" << video_size_.first << "/" << video_size_.second;
        first = false;
    }
    if (!pix_fmt_.empty()) {
        desc << (first ? "=" : ":") << "pix_fmt=" << pix_fmt_;
        first = false;
    }
    if (pixel_aspect_.first != 0 || pixel_aspect_.second != 1) {
        desc << (first ? "=" : ":") << "pixel_aspect=" << pixel_aspect_.first << "/" << pixel_aspect_.second;
        first = false;
    }
    if (frame_rate_.first != 0 || frame_rate_.second != 1) {
        desc << (first ? "=" : ":") << "frame_rate=" << frame_rate_.first << "/" << frame_rate_.second;
        first = false;
    }
    if (colorspace_ != 2) {
        desc << (first ? "=" : ":") << "colorspace=" << colorspace_;
        first = false;
    }
    if (range_ != 0) {
        desc << (first ? "=" : ":") << "range=" << range_;
        first = false;
    }

    return desc.str();
}
