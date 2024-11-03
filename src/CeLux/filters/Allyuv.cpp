#include "Allyuv.hpp"
#include <sstream>

Allyuv::Allyuv(std::pair<int, int> rate, int64_t duration, std::pair<int, int> sar) {
    // Initialize member variables from parameters
    this->rate_ = rate;
    this->duration_ = duration;
    this->sar_ = sar;
}

Allyuv::~Allyuv() {
    // Destructor implementation (if needed)
}

void Allyuv::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Allyuv::getRate() const {
    return rate_;
}

void Allyuv::setDuration(int64_t value) {
    duration_ = value;
}

int64_t Allyuv::getDuration() const {
    return duration_;
}

void Allyuv::setSar(const std::pair<int, int>& value) {
    sar_ = value;
}

std::pair<int, int> Allyuv::getSar() const {
    return sar_;
}

std::string Allyuv::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "allyuv";

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
