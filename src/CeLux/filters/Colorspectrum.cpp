#include "Colorspectrum.hpp"
#include <sstream>

Colorspectrum::Colorspectrum(std::pair<int, int> size, std::pair<int, int> rate, int64_t duration, std::pair<int, int> sar, int type) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->rate_ = rate;
    this->duration_ = duration;
    this->sar_ = sar;
    this->type_ = type;
}

Colorspectrum::~Colorspectrum() {
    // Destructor implementation (if needed)
}

void Colorspectrum::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Colorspectrum::getSize() const {
    return size_;
}

void Colorspectrum::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Colorspectrum::getRate() const {
    return rate_;
}

void Colorspectrum::setDuration(int64_t value) {
    duration_ = value;
}

int64_t Colorspectrum::getDuration() const {
    return duration_;
}

void Colorspectrum::setSar(const std::pair<int, int>& value) {
    sar_ = value;
}

std::pair<int, int> Colorspectrum::getSar() const {
    return sar_;
}

void Colorspectrum::setType(int value) {
    type_ = value;
}

int Colorspectrum::getType() const {
    return type_;
}

std::string Colorspectrum::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "colorspectrum";

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
    if (type_ != 0) {
        desc << (first ? "=" : ":") << "type=" << type_;
        first = false;
    }

    return desc.str();
}
