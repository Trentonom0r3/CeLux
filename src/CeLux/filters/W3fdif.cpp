#include "W3fdif.hpp"
#include <sstream>

W3fdif::W3fdif(int filter, int mode, int parity, int deint) {
    // Initialize member variables from parameters
    this->filter_ = filter;
    this->mode_ = mode;
    this->parity_ = parity;
    this->deint_ = deint;
}

W3fdif::~W3fdif() {
    // Destructor implementation (if needed)
}

void W3fdif::setFilter(int value) {
    filter_ = value;
}

int W3fdif::getFilter() const {
    return filter_;
}

void W3fdif::setMode(int value) {
    mode_ = value;
}

int W3fdif::getMode() const {
    return mode_;
}

void W3fdif::setParity(int value) {
    parity_ = value;
}

int W3fdif::getParity() const {
    return parity_;
}

void W3fdif::setDeint(int value) {
    deint_ = value;
}

int W3fdif::getDeint() const {
    return deint_;
}

std::string W3fdif::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "w3fdif";

    bool first = true;

    if (filter_ != 1) {
        desc << (first ? "=" : ":") << "filter=" << filter_;
        first = false;
    }
    if (mode_ != 1) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (parity_ != -1) {
        desc << (first ? "=" : ":") << "parity=" << parity_;
        first = false;
    }
    if (deint_ != 0) {
        desc << (first ? "=" : ":") << "deint=" << deint_;
        first = false;
    }

    return desc.str();
}
