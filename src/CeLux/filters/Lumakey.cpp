#include "Lumakey.hpp"
#include <sstream>

Lumakey::Lumakey(double threshold, double tolerance, double softness) {
    // Initialize member variables from parameters
    this->threshold_ = threshold;
    this->tolerance_ = tolerance;
    this->softness_ = softness;
}

Lumakey::~Lumakey() {
    // Destructor implementation (if needed)
}

void Lumakey::setThreshold(double value) {
    threshold_ = value;
}

double Lumakey::getThreshold() const {
    return threshold_;
}

void Lumakey::setTolerance(double value) {
    tolerance_ = value;
}

double Lumakey::getTolerance() const {
    return tolerance_;
}

void Lumakey::setSoftness(double value) {
    softness_ = value;
}

double Lumakey::getSoftness() const {
    return softness_;
}

std::string Lumakey::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "lumakey";

    bool first = true;

    if (threshold_ != 0.00) {
        desc << (first ? "=" : ":") << "threshold=" << threshold_;
        first = false;
    }
    if (tolerance_ != 0.01) {
        desc << (first ? "=" : ":") << "tolerance=" << tolerance_;
        first = false;
    }
    if (softness_ != 0.00) {
        desc << (first ? "=" : ":") << "softness=" << softness_;
        first = false;
    }

    return desc.str();
}
