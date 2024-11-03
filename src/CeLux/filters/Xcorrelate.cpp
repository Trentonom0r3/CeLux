#include "Xcorrelate.hpp"
#include <sstream>

Xcorrelate::Xcorrelate(int planes, int secondary) {
    // Initialize member variables from parameters
    this->planes_ = planes;
    this->secondary_ = secondary;
}

Xcorrelate::~Xcorrelate() {
    // Destructor implementation (if needed)
}

void Xcorrelate::setPlanes(int value) {
    planes_ = value;
}

int Xcorrelate::getPlanes() const {
    return planes_;
}

void Xcorrelate::setSecondary(int value) {
    secondary_ = value;
}

int Xcorrelate::getSecondary() const {
    return secondary_;
}

std::string Xcorrelate::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "xcorrelate";

    bool first = true;

    if (planes_ != 7) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }
    if (secondary_ != 1) {
        desc << (first ? "=" : ":") << "secondary=" << secondary_;
        first = false;
    }

    return desc.str();
}
