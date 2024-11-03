#include "Hsvkey.hpp"
#include <sstream>

Hsvkey::Hsvkey(float hue, float sat, float val, float similarity, float blend) {
    // Initialize member variables from parameters
    this->hue_ = hue;
    this->sat_ = sat;
    this->val_ = val;
    this->similarity_ = similarity;
    this->blend_ = blend;
}

Hsvkey::~Hsvkey() {
    // Destructor implementation (if needed)
}

void Hsvkey::setHue(float value) {
    hue_ = value;
}

float Hsvkey::getHue() const {
    return hue_;
}

void Hsvkey::setSat(float value) {
    sat_ = value;
}

float Hsvkey::getSat() const {
    return sat_;
}

void Hsvkey::setVal(float value) {
    val_ = value;
}

float Hsvkey::getVal() const {
    return val_;
}

void Hsvkey::setSimilarity(float value) {
    similarity_ = value;
}

float Hsvkey::getSimilarity() const {
    return similarity_;
}

void Hsvkey::setBlend(float value) {
    blend_ = value;
}

float Hsvkey::getBlend() const {
    return blend_;
}

std::string Hsvkey::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "hsvkey";

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
