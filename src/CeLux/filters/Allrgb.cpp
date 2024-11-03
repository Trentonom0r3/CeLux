#include "Allrgb.hpp"
#include <sstream>

Allrgb::Allrgb(std::pair<int, int> rate, int64_t duration, std::pair<int, int> sar) {
    // Initialize member variables from parameters
    this->rate_ = rate;
    this->duration_ = duration;
    this->sar_ = sar;
}

Allrgb::~Allrgb() {
    // Destructor implementation (if needed)
}

void Allrgb::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Allrgb::getRate() const {
    return rate_;
}

void Allrgb::setDuration(int64_t value) {
    duration_ = value;
}

int64_t Allrgb::getDuration() const {
    return duration_;
}

void Allrgb::setSar(const std::pair<int, int>& value) {
    sar_ = value;
}

std::pair<int, int> Allrgb::getSar() const {
    return sar_;
}

std::string Allrgb::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "allrgb";

    bool first = true;

    if (rate_.first != 0 || rate_.second != 1) {
        desc << (first ? "=" : ":") << "rate=" << rate_.first << "/" << rate_.second;
        first = false;
    }
    if (duration_ != 0) {
        desc << (first ? "=" : ":") << "duration=" << duration_;
        first = false;
    }
    if (sar_.first != 0 || sar_.second != 1) {
        desc << (first ? "=" : ":") << "sar=" << sar_.first << "/" << sar_.second;
        first = false;
    }

    return desc.str();
}
