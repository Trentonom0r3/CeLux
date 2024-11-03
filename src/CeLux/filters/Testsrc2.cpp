#include "Testsrc2.hpp"
#include <sstream>

Testsrc2::Testsrc2(std::pair<int, int> size, std::pair<int, int> rate, int64_t duration, std::pair<int, int> sar, int alpha) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->rate_ = rate;
    this->duration_ = duration;
    this->sar_ = sar;
    this->alpha_ = alpha;
}

Testsrc2::~Testsrc2() {
    // Destructor implementation (if needed)
}

void Testsrc2::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Testsrc2::getSize() const {
    return size_;
}

void Testsrc2::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Testsrc2::getRate() const {
    return rate_;
}

void Testsrc2::setDuration(int64_t value) {
    duration_ = value;
}

int64_t Testsrc2::getDuration() const {
    return duration_;
}

void Testsrc2::setSar(const std::pair<int, int>& value) {
    sar_ = value;
}

std::pair<int, int> Testsrc2::getSar() const {
    return sar_;
}

void Testsrc2::setAlpha(int value) {
    alpha_ = value;
}

int Testsrc2::getAlpha() const {
    return alpha_;
}

std::string Testsrc2::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "testsrc2";

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
    if (alpha_ != 255) {
        desc << (first ? "=" : ":") << "alpha=" << alpha_;
        first = false;
    }

    return desc.str();
}
