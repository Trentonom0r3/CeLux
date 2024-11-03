#include "Nullsrc.hpp"
#include <sstream>

Nullsrc::Nullsrc(std::pair<int, int> size, std::pair<int, int> rate, int64_t duration, std::pair<int, int> sar) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->rate_ = rate;
    this->duration_ = duration;
    this->sar_ = sar;
}

Nullsrc::~Nullsrc() {
    // Destructor implementation (if needed)
}

void Nullsrc::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Nullsrc::getSize() const {
    return size_;
}

void Nullsrc::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Nullsrc::getRate() const {
    return rate_;
}

void Nullsrc::setDuration(int64_t value) {
    duration_ = value;
}

int64_t Nullsrc::getDuration() const {
    return duration_;
}

void Nullsrc::setSar(const std::pair<int, int>& value) {
    sar_ = value;
}

std::pair<int, int> Nullsrc::getSar() const {
    return sar_;
}

std::string Nullsrc::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "nullsrc";

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
