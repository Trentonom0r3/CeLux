#include "Dilation.hpp"
#include <sstream>

Dilation::Dilation(int coordinates, int threshold0, int threshold1, int threshold2, int threshold3) {
    // Initialize member variables from parameters
    this->coordinates_ = coordinates;
    this->threshold0_ = threshold0;
    this->threshold1_ = threshold1;
    this->threshold2_ = threshold2;
    this->threshold3_ = threshold3;
}

Dilation::~Dilation() {
    // Destructor implementation (if needed)
}

void Dilation::setCoordinates(int value) {
    coordinates_ = value;
}

int Dilation::getCoordinates() const {
    return coordinates_;
}

void Dilation::setThreshold0(int value) {
    threshold0_ = value;
}

int Dilation::getThreshold0() const {
    return threshold0_;
}

void Dilation::setThreshold1(int value) {
    threshold1_ = value;
}

int Dilation::getThreshold1() const {
    return threshold1_;
}

void Dilation::setThreshold2(int value) {
    threshold2_ = value;
}

int Dilation::getThreshold2() const {
    return threshold2_;
}

void Dilation::setThreshold3(int value) {
    threshold3_ = value;
}

int Dilation::getThreshold3() const {
    return threshold3_;
}

std::string Dilation::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "dilation";

    bool first = true;

    if (coordinates_ != 255) {
        desc << (first ? "=" : ":") << "coordinates=" << coordinates_;
        first = false;
    }
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
