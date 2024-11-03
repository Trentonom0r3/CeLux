#include "Hsvhold.hpp"
#include <sstream>

Hsvhold::Hsvhold(float hue, float sat, float val, float similarity, float blend) {
    // Initialize member variables from parameters
    this->hue_ = hue;
    this->sat_ = sat;
    this->val_ = val;
    this->similarity_ = similarity;
    this->blend_ = blend;
}

Hsvhold::~Hsvhold() {
    // Destructor implementation (if needed)
}

void Hsvhold::setHue(float value) {
    hue_ = value;
}

float Hsvhold::getHue() const {
    return hue_;
}

void Hsvhold::setSat(float value) {
    sat_ = value;
}

float Hsvhold::getSat() const {
    return sat_;
}

void Hsvhold::setVal(float value) {
    val_ = value;
}

float Hsvhold::getVal() const {
    return val_;
}

void Hsvhold::setSimilarity(float value) {
    similarity_ = value;
}

float Hsvhold::getSimilarity() const {
    return similarity_;
}

void Hsvhold::setBlend(float value) {
    blend_ = value;
}

float Hsvhold::getBlend() const {
    return blend_;
}

std::string Hsvhold::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "hsvhold";

    bool first = true;

    if (hue_ != 0.00) {
        desc << (first ? "=" : ":") << "hue=" << hue_;
        first = false;
    }
    if (sat_ != 0.00) {
        desc << (first ? "=" : ":") << "sat=" << sat_;
        first = false;
    }
    if (val_ != 0.00) {
        desc << (first ? "=" : ":") << "val=" << val_;
        first = false;
    }
    if (similarity_ != 0.01) {
        desc << (first ? "=" : ":") << "similarity=" << similarity_;
        first = false;
    }
    if (blend_ != 0.00) {
        desc << (first ? "=" : ":") << "blend=" << blend_;
        first = false;
    }

    return desc.str();
}
