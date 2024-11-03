#include "Haldclut.hpp"
#include <sstream>

Haldclut::Haldclut(int clut, int interp) {
    // Initialize member variables from parameters
    this->clut_ = clut;
    this->interp_ = interp;
}

Haldclut::~Haldclut() {
    // Destructor implementation (if needed)
}

void Haldclut::setClut(int value) {
    clut_ = value;
}

int Haldclut::getClut() const {
    return clut_;
}

void Haldclut::setInterp(int value) {
    interp_ = value;
}

int Haldclut::getInterp() const {
    return interp_;
}

std::string Haldclut::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "haldclut";

    bool first = true;

    if (clut_ != 1) {
        desc << (first ? "=" : ":") << "clut=" << clut_;
        first = false;
    }
    if (interp_ != 2) {
        desc << (first ? "=" : ":") << "interp=" << interp_;
        first = false;
    }

    return desc.str();
}
