#include "Lut1d.hpp"
#include <sstream>

Lut1d::Lut1d(const std::string& file, int interp) {
    // Initialize member variables from parameters
    this->file_ = file;
    this->interp_ = interp;
}

Lut1d::~Lut1d() {
    // Destructor implementation (if needed)
}

void Lut1d::setFile(const std::string& value) {
    file_ = value;
}

std::string Lut1d::getFile() const {
    return file_;
}

void Lut1d::setInterp(int value) {
    interp_ = value;
}

int Lut1d::getInterp() const {
    return interp_;
}

std::string Lut1d::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "lut1d";

    bool first = true;

    if (!file_.empty()) {
        desc << (first ? "=" : ":") << "file=" << file_;
        first = false;
    }
    if (interp_ != 1) {
        desc << (first ? "=" : ":") << "interp=" << interp_;
        first = false;
    }

    return desc.str();
}
