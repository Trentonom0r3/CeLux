#include "Graphmonitor.hpp"
#include <sstream>

Graphmonitor::Graphmonitor(std::pair<int, int> size, float opacity, int mode, int flags, std::pair<int, int> rate) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->opacity_ = opacity;
    this->mode_ = mode;
    this->flags_ = flags;
    this->rate_ = rate;
}

Graphmonitor::~Graphmonitor() {
    // Destructor implementation (if needed)
}

void Graphmonitor::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Graphmonitor::getSize() const {
    return size_;
}

void Graphmonitor::setOpacity(float value) {
    opacity_ = value;
}

float Graphmonitor::getOpacity() const {
    return opacity_;
}

void Graphmonitor::setMode(int value) {
    mode_ = value;
}

int Graphmonitor::getMode() const {
    return mode_;
}

void Graphmonitor::setFlags(int value) {
    flags_ = value;
}

int Graphmonitor::getFlags() const {
    return flags_;
}

void Graphmonitor::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Graphmonitor::getRate() const {
    return rate_;
}

std::string Graphmonitor::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "graphmonitor";

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
