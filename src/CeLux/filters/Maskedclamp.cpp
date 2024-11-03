#include "Maskedclamp.hpp"
#include <sstream>

Maskedclamp::Maskedclamp(int undershoot, int overshoot, int planes) {
    // Initialize member variables from parameters
    this->undershoot_ = undershoot;
    this->overshoot_ = overshoot;
    this->planes_ = planes;
}

Maskedclamp::~Maskedclamp() {
    // Destructor implementation (if needed)
}

void Maskedclamp::setUndershoot(int value) {
    undershoot_ = value;
}

int Maskedclamp::getUndershoot() const {
    return undershoot_;
}

void Maskedclamp::setOvershoot(int value) {
    overshoot_ = value;
}

int Maskedclamp::getOvershoot() const {
    return overshoot_;
}

void Maskedclamp::setPlanes(int value) {
    planes_ = value;
}

int Maskedclamp::getPlanes() const {
    return planes_;
}

std::string Maskedclamp::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "maskedclamp";

    bool first = true;

    if (undershoot_ != 0) {
        desc << (first ? "=" : ":") << "undershoot=" << undershoot_;
        first = false;
    }
    if (overshoot_ != 0) {
        desc << (first ? "=" : ":") << "overshoot=" << overshoot_;
        first = false;
    }
    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
