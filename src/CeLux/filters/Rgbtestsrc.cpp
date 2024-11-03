#include "Rgbtestsrc.hpp"
#include <sstream>

Rgbtestsrc::Rgbtestsrc(std::pair<int, int> size, std::pair<int, int> rate, int64_t duration, std::pair<int, int> sar, bool complement) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->rate_ = rate;
    this->duration_ = duration;
    this->sar_ = sar;
    this->complement_ = complement;
}

Rgbtestsrc::~Rgbtestsrc() {
    // Destructor implementation (if needed)
}

void Rgbtestsrc::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Rgbtestsrc::getSize() const {
    return size_;
}

void Rgbtestsrc::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Rgbtestsrc::getRate() const {
    return rate_;
}

void Rgbtestsrc::setDuration(int64_t value) {
    duration_ = value;
}

int64_t Rgbtestsrc::getDuration() const {
    return duration_;
}

void Rgbtestsrc::setSar(const std::pair<int, int>& value) {
    sar_ = value;
}

std::pair<int, int> Rgbtestsrc::getSar() const {
    return sar_;
}

void Rgbtestsrc::setComplement(bool value) {
    complement_ = value;
}

bool Rgbtestsrc::getComplement() const {
    return complement_;
}

std::string Rgbtestsrc::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "rgbtestsrc";

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
    if (complement_ != false) {
        desc << (first ? "=" : ":") << "complement=" << (complement_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
