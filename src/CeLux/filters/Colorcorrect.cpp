#include "Colorcorrect.hpp"
#include <sstream>

Colorcorrect::Colorcorrect(float rl, float bl, float rh, float bh, float saturation, int analyze) {
    // Initialize member variables from parameters
    this->rl_ = rl;
    this->bl_ = bl;
    this->rh_ = rh;
    this->bh_ = bh;
    this->saturation_ = saturation;
    this->analyze_ = analyze;
}

Colorcorrect::~Colorcorrect() {
    // Destructor implementation (if needed)
}

void Colorcorrect::setRl(float value) {
    rl_ = value;
}

float Colorcorrect::getRl() const {
    return rl_;
}

void Colorcorrect::setBl(float value) {
    bl_ = value;
}

float Colorcorrect::getBl() const {
    return bl_;
}

void Colorcorrect::setRh(float value) {
    rh_ = value;
}

float Colorcorrect::getRh() const {
    return rh_;
}

void Colorcorrect::setBh(float value) {
    bh_ = value;
}

float Colorcorrect::getBh() const {
    return bh_;
}

void Colorcorrect::setSaturation(float value) {
    saturation_ = value;
}

float Colorcorrect::getSaturation() const {
    return saturation_;
}

void Colorcorrect::setAnalyze(int value) {
    analyze_ = value;
}

int Colorcorrect::getAnalyze() const {
    return analyze_;
}

std::string Colorcorrect::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "colorcorrect";

    bool first = true;

    if (rl_ != 0.00) {
        desc << (first ? "=" : ":") << "rl=" << rl_;
        first = false;
    }
    if (bl_ != 0.00) {
        desc << (first ? "=" : ":") << "bl=" << bl_;
        first = false;
    }
    if (rh_ != 0.00) {
        desc << (first ? "=" : ":") << "rh=" << rh_;
        first = false;
    }
    if (bh_ != 0.00) {
        desc << (first ? "=" : ":") << "bh=" << bh_;
        first = false;
    }
    if (saturation_ != 1.00) {
        desc << (first ? "=" : ":") << "saturation=" << saturation_;
        first = false;
    }
    if (analyze_ != 0) {
        desc << (first ? "=" : ":") << "analyze=" << analyze_;
        first = false;
    }

    return desc.str();
}
