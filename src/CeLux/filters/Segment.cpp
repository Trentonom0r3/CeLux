#include "Segment.hpp"
#include <sstream>

Segment::Segment(const std::string& timestamps, const std::string& frames) {
    // Initialize member variables from parameters
    this->timestamps_ = timestamps;
    this->frames_ = frames;
}

Segment::~Segment() {
    // Destructor implementation (if needed)
}

void Segment::setTimestamps(const std::string& value) {
    timestamps_ = value;
}

std::string Segment::getTimestamps() const {
    return timestamps_;
}

void Segment::setFrames(const std::string& value) {
    frames_ = value;
}

std::string Segment::getFrames() const {
    return frames_;
}

std::string Segment::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "segment";

    bool first = true;

    if (!timestamps_.empty()) {
        desc << (first ? "=" : ":") << "timestamps=" << timestamps_;
        first = false;
    }
    if (!frames_.empty()) {
        desc << (first ? "=" : ":") << "frames=" << frames_;
        first = false;
    }

    return desc.str();
}
