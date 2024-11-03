#include "Fillborders.hpp"
#include <sstream>

Fillborders::Fillborders(int left, int right, int top, int bottom, int mode, const std::string& color) {
    // Initialize member variables from parameters
    this->left_ = left;
    this->right_ = right;
    this->top_ = top;
    this->bottom_ = bottom;
    this->mode_ = mode;
    this->color_ = color;
}

Fillborders::~Fillborders() {
    // Destructor implementation (if needed)
}

void Fillborders::setLeft(int value) {
    left_ = value;
}

int Fillborders::getLeft() const {
    return left_;
}

void Fillborders::setRight(int value) {
    right_ = value;
}

int Fillborders::getRight() const {
    return right_;
}

void Fillborders::setTop(int value) {
    top_ = value;
}

int Fillborders::getTop() const {
    return top_;
}

void Fillborders::setBottom(int value) {
    bottom_ = value;
}

int Fillborders::getBottom() const {
    return bottom_;
}

void Fillborders::setMode(int value) {
    mode_ = value;
}

int Fillborders::getMode() const {
    return mode_;
}

void Fillborders::setColor(const std::string& value) {
    color_ = value;
}

std::string Fillborders::getColor() const {
    return color_;
}

std::string Fillborders::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "fillborders";

    bool first = true;

    if (left_ != 0) {
        desc << (first ? "=" : ":") << "left=" << left_;
        first = false;
    }
    if (right_ != 0) {
        desc << (first ? "=" : ":") << "right=" << right_;
        first = false;
    }
    if (top_ != 0) {
        desc << (first ? "=" : ":") << "top=" << top_;
        first = false;
    }
    if (bottom_ != 0) {
        desc << (first ? "=" : ":") << "bottom=" << bottom_;
        first = false;
    }
    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (color_ != "black") {
        desc << (first ? "=" : ":") << "color=" << color_;
        first = false;
    }

    return desc.str();
}
