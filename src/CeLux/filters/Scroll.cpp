#include "Scroll.hpp"
#include <sstream>

Scroll::Scroll(float horizontal, float vertical, float hpos, float vpos) {
    // Initialize member variables from parameters
    this->horizontal_ = horizontal;
    this->vertical_ = vertical;
    this->hpos_ = hpos;
    this->vpos_ = vpos;
}

Scroll::~Scroll() {
    // Destructor implementation (if needed)
}

void Scroll::setHorizontal(float value) {
    horizontal_ = value;
}

float Scroll::getHorizontal() const {
    return horizontal_;
}

void Scroll::setVertical(float value) {
    vertical_ = value;
}

float Scroll::getVertical() const {
    return vertical_;
}

void Scroll::setHpos(float value) {
    hpos_ = value;
}

float Scroll::getHpos() const {
    return hpos_;
}

void Scroll::setVpos(float value) {
    vpos_ = value;
}

float Scroll::getVpos() const {
    return vpos_;
}

std::string Scroll::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "scroll";

    bool first = true;

    if (horizontal_ != 0.00) {
        desc << (first ? "=" : ":") << "horizontal=" << horizontal_;
        first = false;
    }
    if (vertical_ != 0.00) {
        desc << (first ? "=" : ":") << "vertical=" << vertical_;
        first = false;
    }
    if (hpos_ != 0.00) {
        desc << (first ? "=" : ":") << "hpos=" << hpos_;
        first = false;
    }
    if (vpos_ != 0.00) {
        desc << (first ? "=" : ":") << "vpos=" << vpos_;
        first = false;
    }

    return desc.str();
}
