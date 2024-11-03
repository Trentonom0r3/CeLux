#include "Transpose.hpp"
#include <sstream>

Transpose::Transpose(int dir, int passthrough) {
    // Initialize member variables from parameters
    this->dir_ = dir;
    this->passthrough_ = passthrough;
}

Transpose::~Transpose() {
    // Destructor implementation (if needed)
}

void Transpose::setDir(int value) {
    dir_ = value;
}

int Transpose::getDir() const {
    return dir_;
}

void Transpose::setPassthrough(int value) {
    passthrough_ = value;
}

int Transpose::getPassthrough() const {
    return passthrough_;
}

std::string Transpose::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "transpose";

    bool first = true;

    if (dir_ != 0) {
        desc << (first ? "=" : ":") << "dir=" << dir_;
        first = false;
    }
    if (passthrough_ != 0) {
        desc << (first ? "=" : ":") << "passthrough=" << passthrough_;
        first = false;
    }

    return desc.str();
}
