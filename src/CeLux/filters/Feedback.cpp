#include "Feedback.hpp"
#include <sstream>

Feedback::Feedback(int topLeftCropPosition, int cropSize) {
    // Initialize member variables from parameters
    this->topLeftCropPosition_ = topLeftCropPosition;
    this->cropSize_ = cropSize;
}

Feedback::~Feedback() {
    // Destructor implementation (if needed)
}

void Feedback::setTopLeftCropPosition(int value) {
    topLeftCropPosition_ = value;
}

int Feedback::getTopLeftCropPosition() const {
    return topLeftCropPosition_;
}

void Feedback::setCropSize(int value) {
    cropSize_ = value;
}

int Feedback::getCropSize() const {
    return cropSize_;
}

std::string Feedback::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "feedback";

    bool first = true;

    if (topLeftCropPosition_ != 0) {
        desc << (first ? "=" : ":") << "y=" << topLeftCropPosition_;
        first = false;
    }
    if (cropSize_ != 0) {
        desc << (first ? "=" : ":") << "h=" << cropSize_;
        first = false;
    }

    return desc.str();
}
