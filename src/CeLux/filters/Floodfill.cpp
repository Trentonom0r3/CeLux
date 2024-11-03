#include "Floodfill.hpp"
#include <sstream>

Floodfill::Floodfill(int pixelXCoordinate, int pixelYCoordinate, int s0, int s1, int s2, int s3, int d0, int d1, int d2, int d3) {
    // Initialize member variables from parameters
    this->pixelXCoordinate_ = pixelXCoordinate;
    this->pixelYCoordinate_ = pixelYCoordinate;
    this->s0_ = s0;
    this->s1_ = s1;
    this->s2_ = s2;
    this->s3_ = s3;
    this->d0_ = d0;
    this->d1_ = d1;
    this->d2_ = d2;
    this->d3_ = d3;
}

Floodfill::~Floodfill() {
    // Destructor implementation (if needed)
}

void Floodfill::setPixelXCoordinate(int value) {
    pixelXCoordinate_ = value;
}

int Floodfill::getPixelXCoordinate() const {
    return pixelXCoordinate_;
}

void Floodfill::setPixelYCoordinate(int value) {
    pixelYCoordinate_ = value;
}

int Floodfill::getPixelYCoordinate() const {
    return pixelYCoordinate_;
}

void Floodfill::setS0(int value) {
    s0_ = value;
}

int Floodfill::getS0() const {
    return s0_;
}

void Floodfill::setS1(int value) {
    s1_ = value;
}

int Floodfill::getS1() const {
    return s1_;
}

void Floodfill::setS2(int value) {
    s2_ = value;
}

int Floodfill::getS2() const {
    return s2_;
}

void Floodfill::setS3(int value) {
    s3_ = value;
}

int Floodfill::getS3() const {
    return s3_;
}

void Floodfill::setD0(int value) {
    d0_ = value;
}

int Floodfill::getD0() const {
    return d0_;
}

void Floodfill::setD1(int value) {
    d1_ = value;
}

int Floodfill::getD1() const {
    return d1_;
}

void Floodfill::setD2(int value) {
    d2_ = value;
}

int Floodfill::getD2() const {
    return d2_;
}

void Floodfill::setD3(int value) {
    d3_ = value;
}

int Floodfill::getD3() const {
    return d3_;
}

std::string Floodfill::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "floodfill";

    bool first = true;

    if (pixelXCoordinate_ != 0) {
        desc << (first ? "=" : ":") << "x=" << pixelXCoordinate_;
        first = false;
    }
    if (pixelYCoordinate_ != 0) {
        desc << (first ? "=" : ":") << "y=" << pixelYCoordinate_;
        first = false;
    }
    if (s0_ != 0) {
        desc << (first ? "=" : ":") << "s0=" << s0_;
        first = false;
    }
    if (s1_ != 0) {
        desc << (first ? "=" : ":") << "s1=" << s1_;
        first = false;
    }
    if (s2_ != 0) {
        desc << (first ? "=" : ":") << "s2=" << s2_;
        first = false;
    }
    if (s3_ != 0) {
        desc << (first ? "=" : ":") << "s3=" << s3_;
        first = false;
    }
    if (d0_ != 0) {
        desc << (first ? "=" : ":") << "d0=" << d0_;
        first = false;
    }
    if (d1_ != 0) {
        desc << (first ? "=" : ":") << "d1=" << d1_;
        first = false;
    }
    if (d2_ != 0) {
        desc << (first ? "=" : ":") << "d2=" << d2_;
        first = false;
    }
    if (d3_ != 0) {
        desc << (first ? "=" : ":") << "d3=" << d3_;
        first = false;
    }

    return desc.str();
}
