#include "Shufflepixels.hpp"
#include <sstream>

Shufflepixels::Shufflepixels(int direction, int mode, int width, int height, int64_t seed) {
    // Initialize member variables from parameters
    this->direction_ = direction;
    this->mode_ = mode;
    this->width_ = width;
    this->height_ = height;
    this->seed_ = seed;
}

Shufflepixels::~Shufflepixels() {
    // Destructor implementation (if needed)
}

void Shufflepixels::setDirection(int value) {
    direction_ = value;
}

int Shufflepixels::getDirection() const {
    return direction_;
}

void Shufflepixels::setMode(int value) {
    mode_ = value;
}

int Shufflepixels::getMode() const {
    return mode_;
}

void Shufflepixels::setWidth(int value) {
    width_ = value;
}

int Shufflepixels::getWidth() const {
    return width_;
}

void Shufflepixels::setHeight(int value) {
    height_ = value;
}

int Shufflepixels::getHeight() const {
    return height_;
}

void Shufflepixels::setSeed(int64_t value) {
    seed_ = value;
}

int64_t Shufflepixels::getSeed() const {
    return seed_;
}

std::string Shufflepixels::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "shufflepixels";

    bool first = true;

    if (direction_ != 0) {
        desc << (first ? "=" : ":") << "direction=" << direction_;
        first = false;
    }
    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (width_ != 10) {
        desc << (first ? "=" : ":") << "width=" << width_;
        first = false;
    }
    if (height_ != 10) {
        desc << (first ? "=" : ":") << "height=" << height_;
        first = false;
    }
    if (seed_ != 0) {
        desc << (first ? "=" : ":") << "seed=" << seed_;
        first = false;
    }

    return desc.str();
}
