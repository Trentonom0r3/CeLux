#include "Readeia608.hpp"
#include <sstream>

Readeia608::Readeia608(int scan_min, int scan_max, float spw, bool chp, bool lp) {
    // Initialize member variables from parameters
    this->scan_min_ = scan_min;
    this->scan_max_ = scan_max;
    this->spw_ = spw;
    this->chp_ = chp;
    this->lp_ = lp;
}

Readeia608::~Readeia608() {
    // Destructor implementation (if needed)
}

void Readeia608::setScan_min(int value) {
    scan_min_ = value;
}

int Readeia608::getScan_min() const {
    return scan_min_;
}

void Readeia608::setScan_max(int value) {
    scan_max_ = value;
}

int Readeia608::getScan_max() const {
    return scan_max_;
}

void Readeia608::setSpw(float value) {
    spw_ = value;
}

float Readeia608::getSpw() const {
    return spw_;
}

void Readeia608::setChp(bool value) {
    chp_ = value;
}

bool Readeia608::getChp() const {
    return chp_;
}

void Readeia608::setLp(bool value) {
    lp_ = value;
}

bool Readeia608::getLp() const {
    return lp_;
}

std::string Readeia608::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "readeia608";

    bool first = true;

    if (scan_min_ != 0) {
        desc << (first ? "=" : ":") << "scan_min=" << scan_min_;
        first = false;
    }
    if (scan_max_ != 29) {
        desc << (first ? "=" : ":") << "scan_max=" << scan_max_;
        first = false;
    }
    if (spw_ != 0.27) {
        desc << (first ? "=" : ":") << "spw=" << spw_;
        first = false;
    }
    if (chp_ != false) {
        desc << (first ? "=" : ":") << "chp=" << (chp_ ? "1" : "0");
        first = false;
    }
    if (lp_ != true) {
        desc << (first ? "=" : ":") << "lp=" << (lp_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
