#include "Remap.hpp"
#include <sstream>

Remap::Remap(int format, const std::string& fill) {
    // Initialize member variables from parameters
    this->format_ = format;
    this->fill_ = fill;
}

Remap::~Remap() {
    // Destructor implementation (if needed)
}

void Remap::setFormat(int value) {
    format_ = value;
}

int Remap::getFormat() const {
    return format_;
}

void Remap::setFill(const std::string& value) {
    fill_ = value;
}

std::string Remap::getFill() const {
    return fill_;
}

std::string Remap::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "remap";

    bool first = true;

    if (format_ != 0) {
        desc << (first ? "=" : ":") << "format=" << format_;
        first = false;
    }
    if (fill_ != "black") {
        desc << (first ? "=" : ":") << "fill=" << fill_;
        first = false;
    }

    return desc.str();
}
