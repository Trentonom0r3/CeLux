#include "Multiply.hpp"
#include <sstream>

Multiply::Multiply(float scale, float offset, int planes) {
    // Initialize member variables from parameters
    this->scale_ = scale;
    this->offset_ = offset;
    this->planes_ = planes;
}

Multiply::~Multiply() {
    // Destructor implementation (if needed)
}

void Multiply::setScale(float value) {
    scale_ = value;
}

float Multiply::getScale() const {
    return scale_;
}

void Multiply::setOffset(float value) {
    offset_ = value;
}

float Multiply::getOffset() const {
    return offset_;
}

void Multiply::setPlanes(int value) {
    planes_ = value;
}

int Multiply::getPlanes() const {
    return planes_;
}

std::string Multiply::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "multiply";

    bool first = true;

    if (scale_ != 1.00) {
        desc << (first ? "=" : ":") << "scale=" << scale_;
        first = false;
    }
    if (offset_ != 0.50) {
        desc << (first ? "=" : ":") << "offset=" << offset_;
        first = false;
    }
    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
