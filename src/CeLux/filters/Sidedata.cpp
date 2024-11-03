#include "Sidedata.hpp"
#include <sstream>

Sidedata::Sidedata(int mode, int type) {
    // Initialize member variables from parameters
    this->mode_ = mode;
    this->type_ = type;
}

Sidedata::~Sidedata() {
    // Destructor implementation (if needed)
}

void Sidedata::setMode(int value) {
    mode_ = value;
}

int Sidedata::getMode() const {
    return mode_;
}

void Sidedata::setType(int value) {
    type_ = value;
}

int Sidedata::getType() const {
    return type_;
}

std::string Sidedata::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "sidedata";

    bool first = true;

    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (type_ != -1) {
        desc << (first ? "=" : ":") << "type=" << type_;
        first = false;
    }

    return desc.str();
}
