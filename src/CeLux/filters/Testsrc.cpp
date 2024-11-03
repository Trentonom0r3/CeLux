#include "Testsrc.hpp"
#include <sstream>

Testsrc::Testsrc(std::pair<int, int> size, std::pair<int, int> rate, int64_t duration, std::pair<int, int> sar, int decimals) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->rate_ = rate;
    this->duration_ = duration;
    this->sar_ = sar;
    this->decimals_ = decimals;
}

Testsrc::~Testsrc() {
    // Destructor implementation (if needed)
}

void Testsrc::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Testsrc::getSize() const {
    return size_;
}

void Testsrc::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Testsrc::getRate() const {
    return rate_;
}

void Testsrc::setDuration(int64_t value) {
    duration_ = value;
}

int64_t Testsrc::getDuration() const {
    return duration_;
}

void Testsrc::setSar(const std::pair<int, int>& value) {
    sar_ = value;
}

std::pair<int, int> Testsrc::getSar() const {
    return sar_;
}

void Testsrc::setDecimals(int value) {
    decimals_ = value;
}

int Testsrc::getDecimals() const {
    return decimals_;
}

std::string Testsrc::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "testsrc";

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
    if (decimals_ != 0) {
        desc << (first ? "=" : ":") << "decimals=" << decimals_;
        first = false;
    }

    return desc.str();
}
