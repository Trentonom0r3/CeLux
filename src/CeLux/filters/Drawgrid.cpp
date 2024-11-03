#include "Drawgrid.hpp"
#include <sstream>

Drawgrid::Drawgrid(const std::string& horizontalOffset, const std::string& verticalOffset, const std::string& width, const std::string& height, const std::string& color, const std::string& thickness, bool replace) {
    // Initialize member variables from parameters
    this->horizontalOffset_ = horizontalOffset;
    this->verticalOffset_ = verticalOffset;
    this->width_ = width;
    this->height_ = height;
    this->color_ = color;
    this->thickness_ = thickness;
    this->replace_ = replace;
}

Drawgrid::~Drawgrid() {
    // Destructor implementation (if needed)
}

void Drawgrid::setHorizontalOffset(const std::string& value) {
    horizontalOffset_ = value;
}

std::string Drawgrid::getHorizontalOffset() const {
    return horizontalOffset_;
}

void Drawgrid::setVerticalOffset(const std::string& value) {
    verticalOffset_ = value;
}

std::string Drawgrid::getVerticalOffset() const {
    return verticalOffset_;
}

void Drawgrid::setWidth(const std::string& value) {
    width_ = value;
}

std::string Drawgrid::getWidth() const {
    return width_;
}

void Drawgrid::setHeight(const std::string& value) {
    height_ = value;
}

std::string Drawgrid::getHeight() const {
    return height_;
}

void Drawgrid::setColor(const std::string& value) {
    color_ = value;
}

std::string Drawgrid::getColor() const {
    return color_;
}

void Drawgrid::setThickness(const std::string& value) {
    thickness_ = value;
}

std::string Drawgrid::getThickness() const {
    return thickness_;
}

void Drawgrid::setReplace(bool value) {
    replace_ = value;
}

bool Drawgrid::getReplace() const {
    return replace_;
}

std::string Drawgrid::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "drawgrid";

    bool first = true;

    if (horizontalOffset_ != "0") {
        desc << (first ? "=" : ":") << "x=" << horizontalOffset_;
        first = false;
    }
    if (verticalOffset_ != "0") {
        desc << (first ? "=" : ":") << "y=" << verticalOffset_;
        first = false;
    }
    if (width_ != "0") {
        desc << (first ? "=" : ":") << "width=" << width_;
        first = false;
    }
    if (height_ != "0") {
        desc << (first ? "=" : ":") << "height=" << height_;
        first = false;
    }
    if (color_ != "black") {
        desc << (first ? "=" : ":") << "color=" << color_;
        first = false;
    }
    if (thickness_ != "1") {
        desc << (first ? "=" : ":") << "thickness=" << thickness_;
        first = false;
    }
    if (replace_ != false) {
        desc << (first ? "=" : ":") << "replace=" << (replace_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
