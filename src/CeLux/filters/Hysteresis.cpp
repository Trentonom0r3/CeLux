#include "Hysteresis.hpp"
#include <sstream>

Hysteresis::Hysteresis(int planes, int threshold) {
    // Initialize member variables from parameters
    this->planes_ = planes;
    this->threshold_ = threshold;
}

Hysteresis::~Hysteresis() {
    // Destructor implementation (if needed)
}

void Hysteresis::setPlanes(int value) {
    planes_ = value;
}

int Hysteresis::getPlanes() const {
    return planes_;
}

void Hysteresis::setThreshold(int value) {
    threshold_ = value;
}

int Hysteresis::getThreshold() const {
    return threshold_;
}

std::string Hysteresis::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "hysteresis";

    bool first = true;

    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }
    if (threshold_ != 0) {
        desc << (first ? "=" : ":") << "threshold=" << threshold_;
        first = false;
    }

    return desc.str();
}
