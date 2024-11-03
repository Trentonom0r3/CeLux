#include "Kirsch.hpp"
#include <sstream>

Kirsch::Kirsch(int planes, float scale, float delta) {
    // Initialize member variables from parameters
    this->planes_ = planes;
    this->scale_ = scale;
    this->delta_ = delta;
}

Kirsch::~Kirsch() {
    // Destructor implementation (if needed)
}

void Kirsch::setPlanes(int value) {
    planes_ = value;
}

int Kirsch::getPlanes() const {
    return planes_;
}

void Kirsch::setScale(float value) {
    scale_ = value;
}

float Kirsch::getScale() const {
    return scale_;
}

void Kirsch::setDelta(float value) {
    delta_ = value;
}

float Kirsch::getDelta() const {
    return delta_;
}

std::string Kirsch::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "kirsch";

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
