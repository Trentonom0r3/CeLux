#include "Freezedetect.hpp"
#include <sstream>

Freezedetect::Freezedetect(double noise, int64_t duration) {
    // Initialize member variables from parameters
    this->noise_ = noise;
    this->duration_ = duration;
}

Freezedetect::~Freezedetect() {
    // Destructor implementation (if needed)
}

void Freezedetect::setNoise(double value) {
    noise_ = value;
}

double Freezedetect::getNoise() const {
    return noise_;
}

void Freezedetect::setDuration(int64_t value) {
    duration_ = value;
}

int64_t Freezedetect::getDuration() const {
    return duration_;
}

std::string Freezedetect::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "freezedetect";

    bool first = true;

    if (noise_ != 0.00) {
        desc << (first ? "=" : ":") << "noise=" << noise_;
        first = false;
    }
    if (duration_ != 2000000ULL) {
        desc << (first ? "=" : ":") << "duration=" << duration_;
        first = false;
    }

    return desc.str();
}
