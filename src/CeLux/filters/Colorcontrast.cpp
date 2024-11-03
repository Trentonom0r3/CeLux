#include "Colorcontrast.hpp"
#include <sstream>

Colorcontrast::Colorcontrast(float rc, float gm, float by, float rcw, float gmw, float byw, float pl) {
    // Initialize member variables from parameters
    this->rc_ = rc;
    this->gm_ = gm;
    this->by_ = by;
    this->rcw_ = rcw;
    this->gmw_ = gmw;
    this->byw_ = byw;
    this->pl_ = pl;
}

Colorcontrast::~Colorcontrast() {
    // Destructor implementation (if needed)
}

void Colorcontrast::setRc(float value) {
    rc_ = value;
}

float Colorcontrast::getRc() const {
    return rc_;
}

void Colorcontrast::setGm(float value) {
    gm_ = value;
}

float Colorcontrast::getGm() const {
    return gm_;
}

void Colorcontrast::setBy(float value) {
    by_ = value;
}

float Colorcontrast::getBy() const {
    return by_;
}

void Colorcontrast::setRcw(float value) {
    rcw_ = value;
}

float Colorcontrast::getRcw() const {
    return rcw_;
}

void Colorcontrast::setGmw(float value) {
    gmw_ = value;
}

float Colorcontrast::getGmw() const {
    return gmw_;
}

void Colorcontrast::setByw(float value) {
    byw_ = value;
}

float Colorcontrast::getByw() const {
    return byw_;
}

void Colorcontrast::setPl(float value) {
    pl_ = value;
}

float Colorcontrast::getPl() const {
    return pl_;
}

std::string Colorcontrast::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "colorcontrast";

    bool first = true;

    if (rc_ != 0.00) {
        desc << (first ? "=" : ":") << "rc=" << rc_;
        first = false;
    }
    if (gm_ != 0.00) {
        desc << (first ? "=" : ":") << "gm=" << gm_;
        first = false;
    }
    if (by_ != 0.00) {
        desc << (first ? "=" : ":") << "by=" << by_;
        first = false;
    }
    if (rcw_ != 0.00) {
        desc << (first ? "=" : ":") << "rcw=" << rcw_;
        first = false;
    }
    if (gmw_ != 0.00) {
        desc << (first ? "=" : ":") << "gmw=" << gmw_;
        first = false;
    }
    if (byw_ != 0.00) {
        desc << (first ? "=" : ":") << "byw=" << byw_;
        first = false;
    }
    if (pl_ != 0.00) {
        desc << (first ? "=" : ":") << "pl=" << pl_;
        first = false;
    }

    return desc.str();
}
