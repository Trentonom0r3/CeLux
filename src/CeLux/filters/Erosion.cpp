#include "Erosion.hpp"
#include <sstream>

Erosion::Erosion(int coordinates, int threshold0, int threshold1, int threshold2, int threshold3) {
    // Initialize member variables from parameters
    this->coordinates_ = coordinates;
    this->threshold0_ = threshold0;
    this->threshold1_ = threshold1;
    this->threshold2_ = threshold2;
    this->threshold3_ = threshold3;
}

Erosion::~Erosion() {
    // Destructor implementation (if needed)
}

void Erosion::setCoordinates(int value) {
    coordinates_ = value;
}

int Erosion::getCoordinates() const {
    return coordinates_;
}

void Erosion::setThreshold0(int value) {
    threshold0_ = value;
}

int Erosion::getThreshold0() const {
    return threshold0_;
}

void Erosion::setThreshold1(int value) {
    threshold1_ = value;
}

int Erosion::getThreshold1() const {
    return threshold1_;
}

void Erosion::setThreshold2(int value) {
    threshold2_ = value;
}

int Erosion::getThreshold2() const {
    return threshold2_;
}

void Erosion::setThreshold3(int value) {
    threshold3_ = value;
}

int Erosion::getThreshold3() const {
    return threshold3_;
}

std::string Erosion::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "erosion";

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
