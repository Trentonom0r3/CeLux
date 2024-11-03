#include "Lut3d.hpp"
#include <sstream>

Lut3d::Lut3d(const std::string& file, int clut, int interp) {
    // Initialize member variables from parameters
    this->file_ = file;
    this->clut_ = clut;
    this->interp_ = interp;
}

Lut3d::~Lut3d() {
    // Destructor implementation (if needed)
}

void Lut3d::setFile(const std::string& value) {
    file_ = value;
}

std::string Lut3d::getFile() const {
    return file_;
}

void Lut3d::setClut(int value) {
    clut_ = value;
}

int Lut3d::getClut() const {
    return clut_;
}

void Lut3d::setInterp(int value) {
    interp_ = value;
}

int Lut3d::getInterp() const {
    return interp_;
}

std::string Lut3d::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "lut3d";

    bool first = true;

    if (!file_.empty()) {
        desc << (first ? "=" : ":") << "file=" << file_;
        first = false;
    }
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
