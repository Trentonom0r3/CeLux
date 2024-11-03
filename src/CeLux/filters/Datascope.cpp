#include "Datascope.hpp"
#include <sstream>

Datascope::Datascope(std::pair<int, int> size, int xOffset, int yOffset, int mode, bool axis, float opacity, int format, int components) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->xOffset_ = xOffset;
    this->yOffset_ = yOffset;
    this->mode_ = mode;
    this->axis_ = axis;
    this->opacity_ = opacity;
    this->format_ = format;
    this->components_ = components;
}

Datascope::~Datascope() {
    // Destructor implementation (if needed)
}

void Datascope::setSize(const std::pair<int, int>& value) {
    size_ = value;
}

std::pair<int, int> Datascope::getSize() const {
    return size_;
}

void Datascope::setXOffset(int value) {
    xOffset_ = value;
}

int Datascope::getXOffset() const {
    return xOffset_;
}

void Datascope::setYOffset(int value) {
    yOffset_ = value;
}

int Datascope::getYOffset() const {
    return yOffset_;
}

void Datascope::setMode(int value) {
    mode_ = value;
}

int Datascope::getMode() const {
    return mode_;
}

void Datascope::setAxis(bool value) {
    axis_ = value;
}

bool Datascope::getAxis() const {
    return axis_;
}

void Datascope::setOpacity(float value) {
    opacity_ = value;
}

float Datascope::getOpacity() const {
    return opacity_;
}

void Datascope::setFormat(int value) {
    format_ = value;
}

int Datascope::getFormat() const {
    return format_;
}

void Datascope::setComponents(int value) {
    components_ = value;
}

int Datascope::getComponents() const {
    return components_;
}

std::string Datascope::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "datascope";

    bool first = true;

    if (size_.first != 0 || size_.second != 1) {
        desc << (first ? "=" : ":") << "size=" << size_.first << "/" << size_.second;
        first = false;
    }
    if (xOffset_ != 0) {
        desc << (first ? "=" : ":") << "x=" << xOffset_;
        first = false;
    }
    if (yOffset_ != 0) {
        desc << (first ? "=" : ":") << "y=" << yOffset_;
        first = false;
    }
    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (axis_ != false) {
        desc << (first ? "=" : ":") << "axis=" << (axis_ ? "1" : "0");
        first = false;
    }
    if (opacity_ != 0.75) {
        desc << (first ? "=" : ":") << "opacity=" << opacity_;
        first = false;
    }
    if (format_ != 0) {
        desc << (first ? "=" : ":") << "format=" << format_;
        first = false;
    }
    if (components_ != 15) {
        desc << (first ? "=" : ":") << "components=" << components_;
        first = false;
    }

    return desc.str();
}
