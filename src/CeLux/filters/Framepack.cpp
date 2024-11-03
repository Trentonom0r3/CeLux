#include "Framepack.hpp"
#include <sstream>

Framepack::Framepack(int format) {
    // Initialize member variables from parameters
    this->format_ = format;
}

Framepack::~Framepack() {
    // Destructor implementation (if needed)
}

void Framepack::setFormat(int value) {
    format_ = value;
}

int Framepack::getFormat() const {
    return format_;
}

std::string Framepack::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "framepack";

    bool first = true;

    if (format_ != 1) {
        desc << (first ? "=" : ":") << "format=" << format_;
        first = false;
    }

    return desc.str();
}
