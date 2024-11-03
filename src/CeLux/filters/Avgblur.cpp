#include "Avgblur.hpp"
#include <sstream>

Avgblur::Avgblur(int sizeX, int planes, int sizeY) {
    // Initialize member variables from parameters
    this->sizeX_ = sizeX;
    this->planes_ = planes;
    this->sizeY_ = sizeY;
}

Avgblur::~Avgblur() {
    // Destructor implementation (if needed)
}

void Avgblur::setSizeX(int value) {
    sizeX_ = value;
}

int Avgblur::getSizeX() const {
    return sizeX_;
}

void Avgblur::setPlanes(int value) {
    planes_ = value;
}

int Avgblur::getPlanes() const {
    return planes_;
}

void Avgblur::setSizeY(int value) {
    sizeY_ = value;
}

int Avgblur::getSizeY() const {
    return sizeY_;
}

std::string Avgblur::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "avgblur";

    bool first = true;

    if (sizeX_ != 1) {
        desc << (first ? "=" : ":") << "sizeX=" << sizeX_;
        first = false;
    }
    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }
    if (sizeY_ != 0) {
        desc << (first ? "=" : ":") << "sizeY=" << sizeY_;
        first = false;
    }

    return desc.str();
}
