#include "Yuvtestsrc.hpp"
#include <sstream>

Yuvtestsrc::Yuvtestsrc(std::pair<int, int> size, std::pair<int, int> rate, int64_t duration, std::pair<int, int> sar) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->rate_ = rate;
    this->duration_ = duration;
    this->sar_ = sar;
}

Yuvtestsrc::~Yuvtestsrc() {
    // Destructor implementation (if needed)
}

void Yuvtestsrc::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Yuvtestsrc::getSize() const {
    return size_;
}

void Yuvtestsrc::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Yuvtestsrc::getRate() const {
    return rate_;
}

void Yuvtestsrc::setDuration(int64_t value) {
    duration_ = value;
}

int64_t Yuvtestsrc::getDuration() const {
    return duration_;
}

void Yuvtestsrc::setSar(const std::pair<int, int>& value) {
    sar_ = value;
}

std::pair<int, int> Yuvtestsrc::getSar() const {
    return sar_;
}

std::string Yuvtestsrc::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "yuvtestsrc";

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
