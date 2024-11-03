#include "Inflate.hpp"
#include <sstream>

Inflate::Inflate(int threshold0, int threshold1, int threshold2, int threshold3) {
    // Initialize member variables from parameters
    this->threshold0_ = threshold0;
    this->threshold1_ = threshold1;
    this->threshold2_ = threshold2;
    this->threshold3_ = threshold3;
}

Inflate::~Inflate() {
    // Destructor implementation (if needed)
}

void Inflate::setThreshold0(int value) {
    threshold0_ = value;
}

int Inflate::getThreshold0() const {
    return threshold0_;
}

void Inflate::setThreshold1(int value) {
    threshold1_ = value;
}

int Inflate::getThreshold1() const {
    return threshold1_;
}

void Inflate::setThreshold2(int value) {
    threshold2_ = value;
}

int Inflate::getThreshold2() const {
    return threshold2_;
}

void Inflate::setThreshold3(int value) {
    threshold3_ = value;
}

int Inflate::getThreshold3() const {
    return threshold3_;
}

std::string Inflate::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "inflate";

    bool first = true;

    if (threshold0_ != 65535) {
        desc << (first ? "=" : ":") << "threshold0=" << threshold0_;
        first = false;
    }
    if (threshold1_ != 65535) {
        desc << (first ? "=" : ":") << "threshold1=" << threshold1_;
        first = false;
    }
    if (threshold2_ != 65535) {
        desc << (first ? "=" : ":") << "threshold2=" << threshold2_;
        first = false;
    }
    if (threshold3_ != 65535) {
        desc << (first ? "=" : ":") << "threshold3=" << threshold3_;
        first = false;
    }

    return desc.str();
}
