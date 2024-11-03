#include "Deflicker.hpp"
#include <sstream>

Deflicker::Deflicker(int size, int mode, bool bypass) {
    // Initialize member variables from parameters
    this->size_ = size;
    this->mode_ = mode;
    this->bypass_ = bypass;
}

Deflicker::~Deflicker() {
    // Destructor implementation (if needed)
}

void Deflicker::setSize(int value) {
    size_ = value;
}

int Deflicker::getSize() const {
    return size_;
}

void Deflicker::setMode(int value) {
    mode_ = value;
}

int Deflicker::getMode() const {
    return mode_;
}

void Deflicker::setBypass(bool value) {
    bypass_ = value;
}

bool Deflicker::getBypass() const {
    return bypass_;
}

std::string Deflicker::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "deflicker";

    bool first = true;

    if (size_ != 5) {
        desc << (first ? "=" : ":") << "size=" << size_;
        first = false;
    }
    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (bypass_ != false) {
        desc << (first ? "=" : ":") << "bypass=" << (bypass_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
