#include "Pad.hpp"
#include <sstream>

Pad::Pad(const std::string& width, const std::string& height, const std::string& xOffsetForTheInputImagePosition, const std::string& yOffsetForTheInputImagePosition, const std::string& color, int eval, std::pair<int, int> aspect) {
    // Initialize member variables from parameters
    this->width_ = width;
    this->height_ = height;
    this->xOffsetForTheInputImagePosition_ = xOffsetForTheInputImagePosition;
    this->yOffsetForTheInputImagePosition_ = yOffsetForTheInputImagePosition;
    this->color_ = color;
    this->eval_ = eval;
    this->aspect_ = aspect;
}

Pad::~Pad() {
    // Destructor implementation (if needed)
}

void Pad::setWidth(const std::string& value) {
    width_ = value;
}

std::string Pad::getWidth() const {
    return width_;
}

void Pad::setHeight(const std::string& value) {
    height_ = value;
}

std::string Pad::getHeight() const {
    return height_;
}

void Pad::setXOffsetForTheInputImagePosition(const std::string& value) {
    xOffsetForTheInputImagePosition_ = value;
}

std::string Pad::getXOffsetForTheInputImagePosition() const {
    return xOffsetForTheInputImagePosition_;
}

void Pad::setYOffsetForTheInputImagePosition(const std::string& value) {
    yOffsetForTheInputImagePosition_ = value;
}

std::string Pad::getYOffsetForTheInputImagePosition() const {
    return yOffsetForTheInputImagePosition_;
}

void Pad::setColor(const std::string& value) {
    color_ = value;
}

std::string Pad::getColor() const {
    return color_;
}

void Pad::setEval(int value) {
    eval_ = value;
}

int Pad::getEval() const {
    return eval_;
}

void Pad::setAspect(const std::pair<int, int>& value) {
    aspect_ = value;
}

std::pair<int, int> Pad::getAspect() const {
    return aspect_;
}

std::string Pad::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "pad";

    bool first = true;

    if (width_ != "iw") {
        desc << (first ? "=" : ":") << "width=" << width_;
        first = false;
    }
    if (height_ != "ih") {
        desc << (first ? "=" : ":") << "height=" << height_;
        first = false;
    }
    if (xOffsetForTheInputImagePosition_ != "0") {
        desc << (first ? "=" : ":") << "x=" << xOffsetForTheInputImagePosition_;
        first = false;
    }
    if (yOffsetForTheInputImagePosition_ != "0") {
        desc << (first ? "=" : ":") << "y=" << yOffsetForTheInputImagePosition_;
        first = false;
    }
    if (color_ != "black") {
        desc << (first ? "=" : ":") << "color=" << color_;
        first = false;
    }
    if (eval_ != 0) {
        desc << (first ? "=" : ":") << "eval=" << eval_;
        first = false;
    }
    if (aspect_.first != 0 || aspect_.second != 1) {
        desc << (first ? "=" : ":") << "aspect=" << aspect_.first << "/" << aspect_.second;
        first = false;
    }

    return desc.str();
}
