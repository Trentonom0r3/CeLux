#include "Scharr.hpp"
#include <sstream>

Scharr::Scharr(int planes, float scale, float delta) {
    // Initialize member variables from parameters
    this->planes_ = planes;
    this->scale_ = scale;
    this->delta_ = delta;
}

Scharr::~Scharr() {
    // Destructor implementation (if needed)
}

void Scharr::setPlanes(int value) {
    planes_ = value;
}

int Scharr::getPlanes() const {
    return planes_;
}

void Scharr::setScale(float value) {
    scale_ = value;
}

float Scharr::getScale() const {
    return scale_;
}

void Scharr::setDelta(float value) {
    delta_ = value;
}

float Scharr::getDelta() const {
    return delta_;
}

std::string Scharr::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "scharr";

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
