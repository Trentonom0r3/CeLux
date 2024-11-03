#include "Hwmap.hpp"
#include <sstream>

Hwmap::Hwmap(int mode, const std::string& derive_device, int reverse) {
    // Initialize member variables from parameters
    this->mode_ = mode;
    this->derive_device_ = derive_device;
    this->reverse_ = reverse;
}

Hwmap::~Hwmap() {
    // Destructor implementation (if needed)
}

void Hwmap::setMode(int value) {
    mode_ = value;
}

int Hwmap::getMode() const {
    return mode_;
}

void Hwmap::setDerive_device(const std::string& value) {
    derive_device_ = value;
}

std::string Hwmap::getDerive_device() const {
    return derive_device_;
}

void Hwmap::setReverse(int value) {
    reverse_ = value;
}

int Hwmap::getReverse() const {
    return reverse_;
}

std::string Hwmap::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "hwmap";

    bool first = true;

    if (mode_ != 3) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (!derive_device_.empty()) {
        desc << (first ? "=" : ":") << "derive_device=" << derive_device_;
        first = false;
    }
    if (reverse_ != 0) {
        desc << (first ? "=" : ":") << "reverse=" << reverse_;
        first = false;
    }

    return desc.str();
}
