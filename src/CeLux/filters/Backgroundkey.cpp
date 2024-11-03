#include "Backgroundkey.hpp"
#include <sstream>

Backgroundkey::Backgroundkey(float threshold, float similarity, float blend) {
    // Initialize member variables from parameters
    this->threshold_ = threshold;
    this->similarity_ = similarity;
    this->blend_ = blend;
}

Backgroundkey::~Backgroundkey() {
    // Destructor implementation (if needed)
}

void Backgroundkey::setThreshold(float value) {
    threshold_ = value;
}

float Backgroundkey::getThreshold() const {
    return threshold_;
}

void Backgroundkey::setSimilarity(float value) {
    similarity_ = value;
}

float Backgroundkey::getSimilarity() const {
    return similarity_;
}

void Backgroundkey::setBlend(float value) {
    blend_ = value;
}

float Backgroundkey::getBlend() const {
    return blend_;
}

std::string Backgroundkey::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "backgroundkey";

    bool first = true;

    if (threshold_ != 0.08) {
        desc << (first ? "=" : ":") << "threshold=" << threshold_;
        first = false;
    }
    if (similarity_ != 0.10) {
        desc << (first ? "=" : ":") << "similarity=" << similarity_;
        first = false;
    }
    if (blend_ != 0.00) {
        desc << (first ? "=" : ":") << "blend=" << blend_;
        first = false;
    }

    return desc.str();
}
