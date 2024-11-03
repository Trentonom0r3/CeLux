#include "Drawbox.hpp"
#include <sstream>

Drawbox::Drawbox(const std::string& horizontalPositionOfTheLeftBoxEdge, const std::string& verticalPositionOfTheTopBoxEdge, const std::string& width, const std::string& height, const std::string& color, const std::string& thickness, bool replace, const std::string& box_source) {
    // Initialize member variables from parameters
    this->horizontalPositionOfTheLeftBoxEdge_ = horizontalPositionOfTheLeftBoxEdge;
    this->verticalPositionOfTheTopBoxEdge_ = verticalPositionOfTheTopBoxEdge;
    this->width_ = width;
    this->height_ = height;
    this->color_ = color;
    this->thickness_ = thickness;
    this->replace_ = replace;
    this->box_source_ = box_source;
}

Drawbox::~Drawbox() {
    // Destructor implementation (if needed)
}

void Drawbox::setHorizontalPositionOfTheLeftBoxEdge(const std::string& value) {
    horizontalPositionOfTheLeftBoxEdge_ = value;
}

std::string Drawbox::getHorizontalPositionOfTheLeftBoxEdge() const {
    return horizontalPositionOfTheLeftBoxEdge_;
}

void Drawbox::setVerticalPositionOfTheTopBoxEdge(const std::string& value) {
    verticalPositionOfTheTopBoxEdge_ = value;
}

std::string Drawbox::getVerticalPositionOfTheTopBoxEdge() const {
    return verticalPositionOfTheTopBoxEdge_;
}

void Drawbox::setWidth(const std::string& value) {
    width_ = value;
}

std::string Drawbox::getWidth() const {
    return width_;
}

void Drawbox::setHeight(const std::string& value) {
    height_ = value;
}

std::string Drawbox::getHeight() const {
    return height_;
}

void Drawbox::setColor(const std::string& value) {
    color_ = value;
}

std::string Drawbox::getColor() const {
    return color_;
}

void Drawbox::setThickness(const std::string& value) {
    thickness_ = value;
}

std::string Drawbox::getThickness() const {
    return thickness_;
}

void Drawbox::setReplace(bool value) {
    replace_ = value;
}

bool Drawbox::getReplace() const {
    return replace_;
}

void Drawbox::setBox_source(const std::string& value) {
    box_source_ = value;
}

std::string Drawbox::getBox_source() const {
    return box_source_;
}

std::string Drawbox::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "drawbox";

    bool first = true;

    if (horizontalPositionOfTheLeftBoxEdge_ != "0") {
        desc << (first ? "=" : ":") << "x=" << horizontalPositionOfTheLeftBoxEdge_;
        first = false;
    }
    if (verticalPositionOfTheTopBoxEdge_ != "0") {
        desc << (first ? "=" : ":") << "y=" << verticalPositionOfTheTopBoxEdge_;
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
    if (thickness_ != "3") {
        desc << (first ? "=" : ":") << "thickness=" << thickness_;
        first = false;
    }
    if (replace_ != false) {
        desc << (first ? "=" : ":") << "replace=" << (replace_ ? "1" : "0");
        first = false;
    }
    if (!box_source_.empty()) {
        desc << (first ? "=" : ":") << "box_source=" << box_source_;
        first = false;
    }

    return desc.str();
}
