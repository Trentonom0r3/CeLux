#include "Lut2.hpp"
#include <sstream>

Lut2::Lut2(const std::string& c0, const std::string& c1, const std::string& c2, const std::string& c3, int outputDepth) {
    // Initialize member variables from parameters
    this->c0_ = c0;
    this->c1_ = c1;
    this->c2_ = c2;
    this->c3_ = c3;
    this->outputDepth_ = outputDepth;
}

Lut2::~Lut2() {
    // Destructor implementation (if needed)
}

void Lut2::setC0(const std::string& value) {
    c0_ = value;
}

std::string Lut2::getC0() const {
    return c0_;
}

void Lut2::setC1(const std::string& value) {
    c1_ = value;
}

std::string Lut2::getC1() const {
    return c1_;
}

void Lut2::setC2(const std::string& value) {
    c2_ = value;
}

std::string Lut2::getC2() const {
    return c2_;
}

void Lut2::setC3(const std::string& value) {
    c3_ = value;
}

std::string Lut2::getC3() const {
    return c3_;
}

void Lut2::setOutputDepth(int value) {
    outputDepth_ = value;
}

int Lut2::getOutputDepth() const {
    return outputDepth_;
}

std::string Lut2::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "lut2";

    bool first = true;

    if (c0_ != "x") {
        desc << (first ? "=" : ":") << "c0=" << c0_;
        first = false;
    }
    if (c1_ != "x") {
        desc << (first ? "=" : ":") << "c1=" << c1_;
        first = false;
    }
    if (c2_ != "x") {
        desc << (first ? "=" : ":") << "c2=" << c2_;
        first = false;
    }
    if (c3_ != "x") {
        desc << (first ? "=" : ":") << "c3=" << c3_;
        first = false;
    }
    if (outputDepth_ != 0) {
        desc << (first ? "=" : ":") << "d=" << outputDepth_;
        first = false;
    }

    return desc.str();
}
