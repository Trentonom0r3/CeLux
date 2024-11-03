#include "Zoompan.hpp"
#include <sstream>

Zoompan::Zoompan(const std::string& zoom, const std::string& x, const std::string& y, const std::string& duration, std::pair<int, int> outputImageSize, std::pair<int, int> fps) {
    // Initialize member variables from parameters
    this->zoom_ = zoom;
    this->x_ = x;
    this->y_ = y;
    this->duration_ = duration;
    this->outputImageSize_ = outputImageSize;
    this->fps_ = fps;
}

Zoompan::~Zoompan() {
    // Destructor implementation (if needed)
}

void Zoompan::setZoom(const std::string& value) {
    zoom_ = value;
}

std::string Zoompan::getZoom() const {
    return zoom_;
}

void Zoompan::setX(const std::string& value) {
    x_ = value;
}

std::string Zoompan::getX() const {
    return x_;
}

void Zoompan::setY(const std::string& value) {
    y_ = value;
}

std::string Zoompan::getY() const {
    return y_;
}

void Zoompan::setDuration(const std::string& value) {
    duration_ = value;
}

std::string Zoompan::getDuration() const {
    return duration_;
}

void Zoompan::setOutputImageSize(const std::pair<int, int>& value) {
    outputImageSize_ = value;
}

std::pair<int, int> Zoompan::getOutputImageSize() const {
    return outputImageSize_;
}

void Zoompan::setFps(const std::pair<int, int>& value) {
    fps_ = value;
}

std::pair<int, int> Zoompan::getFps() const {
    return fps_;
}

std::string Zoompan::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "zoompan";

    bool first = true;

    if (zoom_ != "1") {
        desc << (first ? "=" : ":") << "zoom=" << zoom_;
        first = false;
    }
    if (x_ != "0") {
        desc << (first ? "=" : ":") << "x=" << x_;
        first = false;
    }
    if (y_ != "0") {
        desc << (first ? "=" : ":") << "y=" << y_;
        first = false;
    }
    if (duration_ != "90") {
        desc << (first ? "=" : ":") << "d=" << duration_;
        first = false;
    }
    if (outputImageSize_.first != 0 || outputImageSize_.second != 1) {
        desc << (first ? "=" : ":") << "s=" << outputImageSize_.first << "/" << outputImageSize_.second;
        first = false;
    }
    if (fps_.first != 0 || fps_.second != 1) {
        desc << (first ? "=" : ":") << "fps=" << fps_.first << "/" << fps_.second;
        first = false;
    }

    return desc.str();
}
