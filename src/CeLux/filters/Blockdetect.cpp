#include "Blockdetect.hpp"
#include <sstream>

Blockdetect::Blockdetect(int period_min, int period_max, int planes) {
    // Initialize member variables from parameters
    this->period_min_ = period_min;
    this->period_max_ = period_max;
    this->planes_ = planes;
}

Blockdetect::~Blockdetect() {
    // Destructor implementation (if needed)
}

void Blockdetect::setPeriod_min(int value) {
    period_min_ = value;
}

int Blockdetect::getPeriod_min() const {
    return period_min_;
}

void Blockdetect::setPeriod_max(int value) {
    period_max_ = value;
}

int Blockdetect::getPeriod_max() const {
    return period_max_;
}

void Blockdetect::setPlanes(int value) {
    planes_ = value;
}

int Blockdetect::getPlanes() const {
    return planes_;
}

std::string Blockdetect::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "blockdetect";

    bool first = true;

    if (period_min_ != 3) {
        desc << (first ? "=" : ":") << "period_min=" << period_min_;
        first = false;
    }
    if (period_max_ != 24) {
        desc << (first ? "=" : ":") << "period_max=" << period_max_;
        first = false;
    }
    if (planes_ != 1) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
