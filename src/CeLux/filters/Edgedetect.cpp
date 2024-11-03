#include "Edgedetect.hpp"
#include <sstream>

Edgedetect::Edgedetect(double high, double low, int mode, int planes) {
    // Initialize member variables from parameters
    this->high_ = high;
    this->low_ = low;
    this->mode_ = mode;
    this->planes_ = planes;
}

Edgedetect::~Edgedetect() {
    // Destructor implementation (if needed)
}

void Edgedetect::setHigh(double value) {
    high_ = value;
}

double Edgedetect::getHigh() const {
    return high_;
}

void Edgedetect::setLow(double value) {
    low_ = value;
}

double Edgedetect::getLow() const {
    return low_;
}

void Edgedetect::setMode(int value) {
    mode_ = value;
}

int Edgedetect::getMode() const {
    return mode_;
}

void Edgedetect::setPlanes(int value) {
    planes_ = value;
}

int Edgedetect::getPlanes() const {
    return planes_;
}

std::string Edgedetect::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "edgedetect";

    bool first = true;

    if (high_ != 0.20) {
        desc << (first ? "=" : ":") << "high=" << high_;
        first = false;
    }
    if (low_ != 0.08) {
        desc << (first ? "=" : ":") << "low=" << low_;
        first = false;
    }
    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (planes_ != 7) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
