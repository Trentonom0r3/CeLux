#include "Smptebars.hpp"
#include <sstream>

Smptebars::Smptebars(std::pair<int, int> size, std::pair<int, int> rate, int64_t duration, std::pair<int, int> sar) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->rate_ = rate;
    this->duration_ = duration;
    this->sar_ = sar;
}

Smptebars::~Smptebars() {
    // Destructor implementation (if needed)
}

void Smptebars::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Smptebars::getSize() const {
    return size_;
}

void Smptebars::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Smptebars::getRate() const {
    return rate_;
}

void Smptebars::setDuration(int64_t value) {
    duration_ = value;
}

int64_t Smptebars::getDuration() const {
    return duration_;
}

void Smptebars::setSar(const std::pair<int, int>& value) {
    sar_ = value;
}

std::pair<int, int> Smptebars::getSar() const {
    return sar_;
}

std::string Smptebars::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "smptebars";

    bool first = true;

    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
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
