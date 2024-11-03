#include "Chromahold.hpp"
#include <sstream>

Chromahold::Chromahold(const std::string& color, float similarity, float blend, bool yuv) {
    // Initialize member variables from parameters
    this->color_ = color;
    this->similarity_ = similarity;
    this->blend_ = blend;
    this->yuv_ = yuv;
}

Chromahold::~Chromahold() {
    // Destructor implementation (if needed)
}

void Chromahold::setColor(const std::string& value) {
    color_ = value;
}

std::string Chromahold::getColor() const {
    return color_;
}

void Chromahold::setSimilarity(float value) {
    similarity_ = value;
}

float Chromahold::getSimilarity() const {
    return similarity_;
}

void Chromahold::setBlend(float value) {
    blend_ = value;
}

float Chromahold::getBlend() const {
    return blend_;
}

void Chromahold::setYuv(bool value) {
    yuv_ = value;
}

bool Chromahold::getYuv() const {
    return yuv_;
}

std::string Chromahold::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "chromahold";

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
    if (yuv_ != false) {
        desc << (first ? "=" : ":") << "yuv=" << (yuv_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
