#include "Pal75bars.hpp"
#include <sstream>

Pal75bars::Pal75bars(std::pair<int, int> size, std::pair<int, int> rate, int64_t duration, std::pair<int, int> sar) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->rate_ = rate;
    this->duration_ = duration;
    this->sar_ = sar;
}

Pal75bars::~Pal75bars() {
    // Destructor implementation (if needed)
}

void Pal75bars::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Pal75bars::getSize() const {
    return size_;
}

void Pal75bars::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Pal75bars::getRate() const {
    return rate_;
}

void Pal75bars::setDuration(int64_t value) {
    duration_ = value;
}

int64_t Pal75bars::getDuration() const {
    return duration_;
}

void Pal75bars::setSar(const std::pair<int, int>& value) {
    sar_ = value;
}

std::pair<int, int> Pal75bars::getSar() const {
    return sar_;
}

std::string Pal75bars::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "pal75bars";

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
