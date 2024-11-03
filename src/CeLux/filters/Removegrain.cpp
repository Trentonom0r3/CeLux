#include "Removegrain.hpp"
#include <sstream>

Removegrain::Removegrain(int m0, int m1, int m2, int m3) {
    // Initialize member variables from parameters
    this->m0_ = m0;
    this->m1_ = m1;
    this->m2_ = m2;
    this->m3_ = m3;
}

Removegrain::~Removegrain() {
    // Destructor implementation (if needed)
}

void Removegrain::setM0(int value) {
    m0_ = value;
}

int Removegrain::getM0() const {
    return m0_;
}

void Removegrain::setM1(int value) {
    m1_ = value;
}

int Removegrain::getM1() const {
    return m1_;
}

void Removegrain::setM2(int value) {
    m2_ = value;
}

int Removegrain::getM2() const {
    return m2_;
}

void Removegrain::setM3(int value) {
    m3_ = value;
}

int Removegrain::getM3() const {
    return m3_;
}

std::string Removegrain::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "removegrain";

    bool first = true;

    if (m0_ != 0) {
        desc << (first ? "=" : ":") << "m0=" << m0_;
        first = false;
    }
    if (m1_ != 0) {
        desc << (first ? "=" : ":") << "m1=" << m1_;
        first = false;
    }
    if (m2_ != 0) {
        desc << (first ? "=" : ":") << "m2=" << m2_;
        first = false;
    }
    if (m3_ != 0) {
        desc << (first ? "=" : ":") << "m3=" << m3_;
        first = false;
    }

    return desc.str();
}
