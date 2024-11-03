#include "Yadif.hpp"
#include <sstream>

Yadif::Yadif(int mode, int parity, int deint) {
    // Initialize member variables from parameters
    this->mode_ = mode;
    this->parity_ = parity;
    this->deint_ = deint;
}

Yadif::~Yadif() {
    // Destructor implementation (if needed)
}

void Yadif::setMode(int value) {
    mode_ = value;
}

int Yadif::getMode() const {
    return mode_;
}

void Yadif::setParity(int value) {
    parity_ = value;
}

int Yadif::getParity() const {
    return parity_;
}

void Yadif::setDeint(int value) {
    deint_ = value;
}

int Yadif::getDeint() const {
    return deint_;
}

std::string Yadif::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "yadif";

    bool first = true;

    if (mode_ != 0) {
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
