#include "Bwdif.hpp"
#include <sstream>

Bwdif::Bwdif(int mode, int parity, int deint) {
    // Initialize member variables from parameters
    this->mode_ = mode;
    this->parity_ = parity;
    this->deint_ = deint;
}

Bwdif::~Bwdif() {
    // Destructor implementation (if needed)
}

void Bwdif::setMode(int value) {
    mode_ = value;
}

int Bwdif::getMode() const {
    return mode_;
}

void Bwdif::setParity(int value) {
    parity_ = value;
}

int Bwdif::getParity() const {
    return parity_;
}

void Bwdif::setDeint(int value) {
    deint_ = value;
}

int Bwdif::getDeint() const {
    return deint_;
}

std::string Bwdif::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "bwdif";

    bool first = true;

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
