#include "Morpho.hpp"
#include <sstream>

Morpho::Morpho(int mode, int planes, int structure) {
    // Initialize member variables from parameters
    this->mode_ = mode;
    this->planes_ = planes;
    this->structure_ = structure;
}

Morpho::~Morpho() {
    // Destructor implementation (if needed)
}

void Morpho::setMode(int value) {
    mode_ = value;
}

int Morpho::getMode() const {
    return mode_;
}

void Morpho::setPlanes(int value) {
    planes_ = value;
}

int Morpho::getPlanes() const {
    return planes_;
}

void Morpho::setStructure(int value) {
    structure_ = value;
}

int Morpho::getStructure() const {
    return structure_;
}

std::string Morpho::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "morpho";

    bool first = true;

    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (planes_ != 7) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }
    if (structure_ != 1) {
        desc << (first ? "=" : ":") << "structure=" << structure_;
        first = false;
    }

    return desc.str();
}
