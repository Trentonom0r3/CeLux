#include "Dctdnoiz.hpp"
#include <sstream>

Dctdnoiz::Dctdnoiz(float sigma, int overlap, const std::string& expr, int blockSizeExpressedInBits) {
    // Initialize member variables from parameters
    this->sigma_ = sigma;
    this->overlap_ = overlap;
    this->expr_ = expr;
    this->blockSizeExpressedInBits_ = blockSizeExpressedInBits;
}

Dctdnoiz::~Dctdnoiz() {
    // Destructor implementation (if needed)
}

void Dctdnoiz::setSigma(float value) {
    sigma_ = value;
}

float Dctdnoiz::getSigma() const {
    return sigma_;
}

void Dctdnoiz::setOverlap(int value) {
    overlap_ = value;
}

int Dctdnoiz::getOverlap() const {
    return overlap_;
}

void Dctdnoiz::setExpr(const std::string& value) {
    expr_ = value;
}

std::string Dctdnoiz::getExpr() const {
    return expr_;
}

void Dctdnoiz::setBlockSizeExpressedInBits(int value) {
    blockSizeExpressedInBits_ = value;
}

int Dctdnoiz::getBlockSizeExpressedInBits() const {
    return blockSizeExpressedInBits_;
}

std::string Dctdnoiz::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "dctdnoiz";

    bool first = true;

    if (sigma_ != 0.00) {
        desc << (first ? "=" : ":") << "sigma=" << sigma_;
        first = false;
    }
    if (overlap_ != -1) {
        desc << (first ? "=" : ":") << "overlap=" << overlap_;
        first = false;
    }
    if (!expr_.empty()) {
        desc << (first ? "=" : ":") << "expr=" << expr_;
        first = false;
    }
    if (blockSizeExpressedInBits_ != 3) {
        desc << (first ? "=" : ":") << "n=" << blockSizeExpressedInBits_;
        first = false;
    }

    return desc.str();
}
