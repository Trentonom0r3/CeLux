#include "Maskedthreshold.hpp"
#include <sstream>

Maskedthreshold::Maskedthreshold(int threshold, int planes, int mode) {
    // Initialize member variables from parameters
    this->threshold_ = threshold;
    this->planes_ = planes;
    this->mode_ = mode;
}

Maskedthreshold::~Maskedthreshold() {
    // Destructor implementation (if needed)
}

void Maskedthreshold::setThreshold(int value) {
    threshold_ = value;
}

int Maskedthreshold::getThreshold() const {
    return threshold_;
}

void Maskedthreshold::setPlanes(int value) {
    planes_ = value;
}

int Maskedthreshold::getPlanes() const {
    return planes_;
}

void Maskedthreshold::setMode(int value) {
    mode_ = value;
}

int Maskedthreshold::getMode() const {
    return mode_;
}

std::string Maskedthreshold::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "maskedthreshold";

    bool first = true;

    if (threshold_ != 1) {
        desc << (first ? "=" : ":") << "threshold=" << threshold_;
        first = false;
    }
    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }
    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }

    return desc.str();
}
