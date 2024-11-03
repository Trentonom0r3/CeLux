#include "Blurdetect.hpp"
#include <sstream>

Blurdetect::Blurdetect(float high, float low, int radius, int block_pct, int block_height, int planes) {
    // Initialize member variables from parameters
    this->high_ = high;
    this->low_ = low;
    this->radius_ = radius;
    this->block_pct_ = block_pct;
    this->block_height_ = block_height;
    this->planes_ = planes;
}

Blurdetect::~Blurdetect() {
    // Destructor implementation (if needed)
}

void Blurdetect::setHigh(float value) {
    high_ = value;
}

float Blurdetect::getHigh() const {
    return high_;
}

void Blurdetect::setLow(float value) {
    low_ = value;
}

float Blurdetect::getLow() const {
    return low_;
}

void Blurdetect::setRadius(int value) {
    radius_ = value;
}

int Blurdetect::getRadius() const {
    return radius_;
}

void Blurdetect::setBlock_pct(int value) {
    block_pct_ = value;
}

int Blurdetect::getBlock_pct() const {
    return block_pct_;
}

void Blurdetect::setBlock_height(int value) {
    block_height_ = value;
}

int Blurdetect::getBlock_height() const {
    return block_height_;
}

void Blurdetect::setPlanes(int value) {
    planes_ = value;
}

int Blurdetect::getPlanes() const {
    return planes_;
}

std::string Blurdetect::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "blurdetect";

    bool first = true;

    if (high_ != 0.12) {
        desc << (first ? "=" : ":") << "high=" << high_;
        first = false;
    }
    if (low_ != 0.06) {
        desc << (first ? "=" : ":") << "low=" << low_;
        first = false;
    }
    if (radius_ != 50) {
        desc << (first ? "=" : ":") << "radius=" << radius_;
        first = false;
    }
    if (block_pct_ != 80) {
        desc << (first ? "=" : ":") << "block_pct=" << block_pct_;
        first = false;
    }
    if (block_height_ != -1) {
        desc << (first ? "=" : ":") << "block_height=" << block_height_;
        first = false;
    }
    if (planes_ != 1) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
