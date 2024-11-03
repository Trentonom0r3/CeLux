#include "Prewitt.hpp"
#include <sstream>

Prewitt::Prewitt(int planes, float scale, float delta) {
    // Initialize member variables from parameters
    this->planes_ = planes;
    this->scale_ = scale;
    this->delta_ = delta;
}

Prewitt::~Prewitt() {
    // Destructor implementation (if needed)
}

void Prewitt::setPlanes(int value) {
    planes_ = value;
}

int Prewitt::getPlanes() const {
    return planes_;
}

void Prewitt::setScale(float value) {
    scale_ = value;
}

float Prewitt::getScale() const {
    return scale_;
}

void Prewitt::setDelta(float value) {
    delta_ = value;
}

float Prewitt::getDelta() const {
    return delta_;
}

std::string Prewitt::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "prewitt";

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
