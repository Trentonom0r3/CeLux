#include "Exposure.hpp"
#include <sstream>

Exposure::Exposure(float exposure, float black) {
    // Initialize member variables from parameters
    this->exposure_ = exposure;
    this->black_ = black;
}

Exposure::~Exposure() {
    // Destructor implementation (if needed)
}

void Exposure::setExposure(float value) {
    exposure_ = value;
}

float Exposure::getExposure() const {
    return exposure_;
}

void Exposure::setBlack(float value) {
    black_ = value;
}

float Exposure::getBlack() const {
    return black_;
}

std::string Exposure::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "exposure";

    bool first = true;

    if (exposure_ != 0.00) {
        desc << (first ? "=" : ":") << "exposure=" << exposure_;
        first = false;
    }
    if (black_ != 0.00) {
        desc << (first ? "=" : ":") << "black=" << black_;
        first = false;
    }

    return desc.str();
}
