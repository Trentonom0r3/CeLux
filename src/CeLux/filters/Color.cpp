#include "Color.hpp"
#include <sstream>

Color::Color(const std::string& color, std::pair<int, int> size, std::pair<int, int> rate, int64_t duration, std::pair<int, int> sar) {
    // Initialize member variables from parameters
    this->color_ = color;
    this->size_ = size;
    this->rate_ = rate;
    this->duration_ = duration;
    this->sar_ = sar;
}

Color::~Color() {
    // Destructor implementation (if needed)
}

void Color::setColor(const std::string& value) {
    color_ = value;
}

std::string Color::getColor() const {
    return color_;
}

void Color::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Color::getSize() const {
    return size_;
}

void Color::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Color::getRate() const {
    return rate_;
}

void Color::setDuration(int64_t value) {
    duration_ = value;
}

int64_t Color::getDuration() const {
    return duration_;
}

void Color::setSar(const std::pair<int, int>& value) {
    sar_ = value;
}

std::pair<int, int> Color::getSar() const {
    return sar_;
}

std::string Color::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "color";

    bool first = true;

    if (color_ != "black") {
        desc << (first ? "=" : ":") << "color=" << color_;
        first = false;
    }
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
