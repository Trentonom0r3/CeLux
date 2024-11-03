#include "Pixelize.hpp"
#include <sstream>

Pixelize::Pixelize(int width, int height, int mode, int planes) {
    // Initialize member variables from parameters
    this->width_ = width;
    this->height_ = height;
    this->mode_ = mode;
    this->planes_ = planes;
}

Pixelize::~Pixelize() {
    // Destructor implementation (if needed)
}

void Pixelize::setWidth(int value) {
    width_ = value;
}

int Pixelize::getWidth() const {
    return width_;
}

void Pixelize::setHeight(int value) {
    height_ = value;
}

int Pixelize::getHeight() const {
    return height_;
}

void Pixelize::setMode(int value) {
    mode_ = value;
}

int Pixelize::getMode() const {
    return mode_;
}

void Pixelize::setPlanes(int value) {
    planes_ = value;
}

int Pixelize::getPlanes() const {
    return planes_;
}

std::string Pixelize::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "pixelize";

    bool first = true;

    if (width_ != 16) {
        desc << (first ? "=" : ":") << "width=" << width_;
        first = false;
    }
    if (height_ != 16) {
        desc << (first ? "=" : ":") << "height=" << height_;
        first = false;
    }
    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
