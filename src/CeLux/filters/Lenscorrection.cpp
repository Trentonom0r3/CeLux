#include "Lenscorrection.hpp"
#include <sstream>

Lenscorrection::Lenscorrection(double cx, double cy, double k1, double k2, int interpolationType, const std::string& fc) {
    // Initialize member variables from parameters
    this->cx_ = cx;
    this->cy_ = cy;
    this->k1_ = k1;
    this->k2_ = k2;
    this->interpolationType_ = interpolationType;
    this->fc_ = fc;
}

Lenscorrection::~Lenscorrection() {
    // Destructor implementation (if needed)
}

void Lenscorrection::setCx(double value) {
    cx_ = value;
}

double Lenscorrection::getCx() const {
    return cx_;
}

void Lenscorrection::setCy(double value) {
    cy_ = value;
}

double Lenscorrection::getCy() const {
    return cy_;
}

void Lenscorrection::setK1(double value) {
    k1_ = value;
}

double Lenscorrection::getK1() const {
    return k1_;
}

void Lenscorrection::setK2(double value) {
    k2_ = value;
}

double Lenscorrection::getK2() const {
    return k2_;
}

void Lenscorrection::setInterpolationType(int value) {
    interpolationType_ = value;
}

int Lenscorrection::getInterpolationType() const {
    return interpolationType_;
}

void Lenscorrection::setFc(const std::string& value) {
    fc_ = value;
}

std::string Lenscorrection::getFc() const {
    return fc_;
}

std::string Lenscorrection::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "lenscorrection";

    bool first = true;

    if (cx_ != 0.50) {
        desc << (first ? "=" : ":") << "cx=" << cx_;
        first = false;
    }
    if (cy_ != 0.50) {
        desc << (first ? "=" : ":") << "cy=" << cy_;
        first = false;
    }
    if (k1_ != 0.00) {
        desc << (first ? "=" : ":") << "k1=" << k1_;
        first = false;
    }
    if (k2_ != 0.00) {
        desc << (first ? "=" : ":") << "k2=" << k2_;
        first = false;
    }
    if (interpolationType_ != 0) {
        desc << (first ? "=" : ":") << "i=" << interpolationType_;
        first = false;
    }
    if (fc_ != "black@0") {
        desc << (first ? "=" : ":") << "fc=" << fc_;
        first = false;
    }

    return desc.str();
}
