#include "Monochrome.hpp"
#include <sstream>

Monochrome::Monochrome(float cb, float cr, float size, float high) {
    // Initialize member variables from parameters
    this->cb_ = cb;
    this->cr_ = cr;
    this->size_ = size;
    this->high_ = high;
}

Monochrome::~Monochrome() {
    // Destructor implementation (if needed)
}

void Monochrome::setCb(float value) {
    cb_ = value;
}

float Monochrome::getCb() const {
    return cb_;
}

void Monochrome::setCr(float value) {
    cr_ = value;
}

float Monochrome::getCr() const {
    return cr_;
}

void Monochrome::setSize(float value) {
    size_ = value;
}

float Monochrome::getSize() const {
    return size_;
}

void Monochrome::setHigh(float value) {
    high_ = value;
}

float Monochrome::getHigh() const {
    return high_;
}

std::string Monochrome::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "monochrome";

    bool first = true;

    if (cb_ != 0.00) {
        desc << (first ? "=" : ":") << "cb=" << cb_;
        first = false;
    }
    if (cr_ != 0.00) {
        desc << (first ? "=" : ":") << "cr=" << cr_;
        first = false;
    }
    if (size_ != 1.00) {
        desc << (first ? "=" : ":") << "size=" << size_;
        first = false;
    }
    if (high_ != 0.00) {
        desc << (first ? "=" : ":") << "high=" << high_;
        first = false;
    }

    return desc.str();
}
