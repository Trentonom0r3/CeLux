#include "Roberts.hpp"
#include <sstream>

Roberts::Roberts(int planes, float scale, float delta) {
    // Initialize member variables from parameters
    this->planes_ = planes;
    this->scale_ = scale;
    this->delta_ = delta;
}

Roberts::~Roberts() {
    // Destructor implementation (if needed)
}

void Roberts::setPlanes(int value) {
    planes_ = value;
}

int Roberts::getPlanes() const {
    return planes_;
}

void Roberts::setScale(float value) {
    scale_ = value;
}

float Roberts::getScale() const {
    return scale_;
}

void Roberts::setDelta(float value) {
    delta_ = value;
}

float Roberts::getDelta() const {
    return delta_;
}

std::string Roberts::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "roberts";

    bool first = true;

    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }
    if (scale_ != 1.00) {
        desc << (first ? "=" : ":") << "scale=" << scale_;
        first = false;
    }
    if (delta_ != 0.00) {
        desc << (first ? "=" : ":") << "delta=" << delta_;
        first = false;
    }

    return desc.str();
}
