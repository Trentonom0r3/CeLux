#include "Showwaves.hpp"
#include <sstream>

Showwaves::Showwaves(std::pair<int, int> size, int mode, std::pair<int, int> howManySamplesToShowInTheSamePoint, std::pair<int, int> rate, bool split_channels, const std::string& colors, int scale, int draw) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->mode_ = mode;
    this->howManySamplesToShowInTheSamePoint_ = howManySamplesToShowInTheSamePoint;
    this->rate_ = rate;
    this->split_channels_ = split_channels;
    this->colors_ = colors;
    this->scale_ = scale;
    this->draw_ = draw;
}

Showwaves::~Showwaves() {
    // Destructor implementation (if needed)
}

void Showwaves::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Showwaves::getSize() const {
    return size_;
}

void Showwaves::setMode(int value) {
    mode_ = value;
}

int Showwaves::getMode() const {
    return mode_;
}

void Showwaves::setHowManySamplesToShowInTheSamePoint(const std::pair<int, int>& value) {
    howManySamplesToShowInTheSamePoint_ = value;
}

std::pair<int, int> Showwaves::getHowManySamplesToShowInTheSamePoint() const {
    return howManySamplesToShowInTheSamePoint_;
}

void Showwaves::setRate(const std::pair<int, int>& value) {
    rate_ = value;
}

std::pair<int, int> Showwaves::getRate() const {
    return rate_;
}

void Showwaves::setSplit_channels(bool value) {
    split_channels_ = value;
}

bool Showwaves::getSplit_channels() const {
    return split_channels_;
}

void Showwaves::setColors(const std::string& value) {
    colors_ = value;
}

std::string Showwaves::getColors() const {
    return colors_;
}

void Showwaves::setScale(int value) {
    scale_ = value;
}

int Showwaves::getScale() const {
    return scale_;
}

void Showwaves::setDraw(int value) {
    draw_ = value;
}

int Showwaves::getDraw() const {
    return draw_;
}

std::string Showwaves::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "showwaves";

    bool first = true;

    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (howManySamplesToShowInTheSamePoint_.first != 0 || howManySamplesToShowInTheSamePoint_.second != 1) {
        desc << (first ? "=" : ":") << "n=" << howManySamplesToShowInTheSamePoint_.first << "/" << howManySamplesToShowInTheSamePoint_.second;
        first = false;
    }
    if (rate_.first != 0 || rate_.second != 1) {
        desc << (first ? "=" : ":") << "rate=" << rate_.first << "/" << rate_.second;
        first = false;
    }
    if (split_channels_ != false) {
        desc << (first ? "=" : ":") << "split_channels=" << (split_channels_ ? "1" : "0");
        first = false;
    }
    if (colors_ != "red|green|blue|yellow|orange|lime|pink|magenta|brown") {
        desc << (first ? "=" : ":") << "colors=" << colors_;
        first = false;
    }
    if (scale_ != 0) {
        desc << (first ? "=" : ":") << "scale=" << scale_;
        first = false;
    }
    if (draw_ != 0) {
        desc << (first ? "=" : ":") << "draw=" << draw_;
        first = false;
    }

    return desc.str();
}
