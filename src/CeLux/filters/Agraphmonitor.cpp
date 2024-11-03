#include "Agraphmonitor.hpp"
#include <sstream>

Agraphmonitor::Agraphmonitor(std::pair<int, int> size, float opacity, int mode, int flags, std::pair<int, int> rate) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->opacity_ = opacity;
    this->mode_ = mode;
    this->flags_ = flags;
    this->rate_ = rate;
}

Agraphmonitor::~Agraphmonitor() {
    // Destructor implementation (if needed)
}

void Agraphmonitor::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Agraphmonitor::getSize() const {
    return size_;
}

void Agraphmonitor::setOpacity(float value) {
    opacity_ = value;
}

float Agraphmonitor::getOpacity() const {
    return opacity_;
}

void Agraphmonitor::setMode(int value) {
    mode_ = value;
}

int Agraphmonitor::getMode() const {
    return mode_;
}

void Agraphmonitor::setFlags(int value) {
    flags_ = value;
}

int Agraphmonitor::getFlags() const {
    return flags_;
}

void Agraphmonitor::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Agraphmonitor::getRate() const {
    return rate_;
}

std::string Agraphmonitor::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "agraphmonitor";

    bool first = true;

    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (opacity_ != 0.90) {
        desc << (first ? "=" : ":") << "opacity=" << opacity_;
        first = false;
    }
    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (flags_ != 1) {
        desc << (first ? "=" : ":") << "flags=" << flags_;
        first = false;
    }
    if (rate_.first != 0 || rate_.second != 1) {
        desc << (first ? "=" : ":") << "rate=" << rate_.first << "/" << rate_.second;
        first = false;
    }

    return desc.str();
}
