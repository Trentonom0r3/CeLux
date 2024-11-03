#include "Colorkey.hpp"
#include <sstream>

Colorkey::Colorkey(const std::string& color, float similarity, float blend) {
    // Initialize member variables from parameters
    this->color_ = color;
    this->similarity_ = similarity;
    this->blend_ = blend;
}

Colorkey::~Colorkey() {
    // Destructor implementation (if needed)
}

void Colorkey::setColor(const std::string& value) {
    color_ = value;
}

std::string Colorkey::getColor() const {
    return color_;
}

void Colorkey::setSimilarity(float value) {
    similarity_ = value;
}

float Colorkey::getSimilarity() const {
    return similarity_;
}

void Colorkey::setBlend(float value) {
    blend_ = value;
}

float Colorkey::getBlend() const {
    return blend_;
}

std::string Colorkey::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "colorkey";

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
