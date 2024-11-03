#include "Colorhold.hpp"
#include <sstream>

Colorhold::Colorhold(const std::string& color, float similarity, float blend) {
    // Initialize member variables from parameters
    this->color_ = color;
    this->similarity_ = similarity;
    this->blend_ = blend;
}

Colorhold::~Colorhold() {
    // Destructor implementation (if needed)
}

void Colorhold::setColor(const std::string& value) {
    color_ = value;
}

std::string Colorhold::getColor() const {
    return color_;
}

void Colorhold::setSimilarity(float value) {
    similarity_ = value;
}

float Colorhold::getSimilarity() const {
    return similarity_;
}

void Colorhold::setBlend(float value) {
    blend_ = value;
}

float Colorhold::getBlend() const {
    return blend_;
}

std::string Colorhold::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "colorhold";

    bool first = true;

    if (color_ != "black") {
        desc << (first ? "=" : ":") << "color=" << color_;
        first = false;
    }
    if (similarity_ != 0.01) {
        desc << (first ? "=" : ":") << "similarity=" << similarity_;
        first = false;
    }
    if (blend_ != 0.00) {
        desc << (first ? "=" : ":") << "blend=" << blend_;
        first = false;
    }

    return desc.str();
}
