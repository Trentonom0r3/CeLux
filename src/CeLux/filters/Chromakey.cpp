#include "Chromakey.hpp"
#include <sstream>

Chromakey::Chromakey(const std::string& color, float similarity, float blend, bool yuv) {
    // Initialize member variables from parameters
    this->color_ = color;
    this->similarity_ = similarity;
    this->blend_ = blend;
    this->yuv_ = yuv;
}

Chromakey::~Chromakey() {
    // Destructor implementation (if needed)
}

void Chromakey::setColor(const std::string& value) {
    color_ = value;
}

std::string Chromakey::getColor() const {
    return color_;
}

void Chromakey::setSimilarity(float value) {
    similarity_ = value;
}

float Chromakey::getSimilarity() const {
    return similarity_;
}

void Chromakey::setBlend(float value) {
    blend_ = value;
}

float Chromakey::getBlend() const {
    return blend_;
}

void Chromakey::setYuv(bool value) {
    yuv_ = value;
}

bool Chromakey::getYuv() const {
    return yuv_;
}

std::string Chromakey::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "chromakey";

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
