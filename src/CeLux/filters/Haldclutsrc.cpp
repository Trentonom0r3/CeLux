#include "Haldclutsrc.hpp"
#include <sstream>

Haldclutsrc::Haldclutsrc(int level, std::pair<int, int> rate, int64_t duration, std::pair<int, int> sar) {
    // Initialize member variables from parameters
    this->level_ = level;
    this->rate_ = rate;
    this->duration_ = duration;
    this->sar_ = sar;
}

Haldclutsrc::~Haldclutsrc() {
    // Destructor implementation (if needed)
}

void Haldclutsrc::setLevel(int value) {
    level_ = value;
}

int Haldclutsrc::getLevel() const {
    return level_;
}

void Haldclutsrc::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Haldclutsrc::getRate() const {
    return rate_;
}

void Haldclutsrc::setDuration(int64_t value) {
    duration_ = value;
}

int64_t Haldclutsrc::getDuration() const {
    return duration_;
}

void Haldclutsrc::setSar(const std::pair<int, int>& value) {
    sar_ = value;
}

std::pair<int, int> Haldclutsrc::getSar() const {
    return sar_;
}

std::string Haldclutsrc::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "haldclutsrc";

    bool first = true;

    if (level_ != 6) {
        desc << (first ? "=" : ":") << "level=" << level_;
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
