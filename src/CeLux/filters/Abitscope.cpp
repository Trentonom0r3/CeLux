#include "Abitscope.hpp"
#include <sstream>

Abitscope::Abitscope(std::pair<int, int> rate, std::pair<int, int> size, const std::string& colors, int mode) {
    // Initialize member variables from parameters
    this->rate_ = rate;
    this->size_ = size;
    this->colors_ = colors;
    this->mode_ = mode;
}

Abitscope::~Abitscope() {
    // Destructor implementation (if needed)
}

void Abitscope::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Abitscope::getRate() const {
    return rate_;
}

void Abitscope::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Abitscope::getSize() const {
    return size_;
}

void Abitscope::setColors(const std::string& value) {
    colors_ = value;
}

std::string Abitscope::getColors() const {
    return colors_;
}

void Abitscope::setMode(int value) {
    mode_ = value;
}

int Abitscope::getMode() const {
    return mode_;
}

std::string Abitscope::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "abitscope";

    bool first = true;

    if (rate_.first != 0 || rate_.second != 1) {
        desc << (first ? "=" : ":") << "rate=" << rate_.first << "/" << rate_.second;
        first = false;
    }
    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (colors_ != "red|green|blue|yellow|orange|lime|pink|magenta|brown") {
        desc << (first ? "=" : ":") << "colors=" << colors_;
        first = false;
    }
    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }

    return desc.str();
}
