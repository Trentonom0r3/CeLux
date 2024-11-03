#include "Showpalette.hpp"
#include <sstream>

Showpalette::Showpalette(int pixelBoxSize) {
    // Initialize member variables from parameters
    this->pixelBoxSize_ = pixelBoxSize;
}

Showpalette::~Showpalette() {
    // Destructor implementation (if needed)
}

void Showpalette::setPixelBoxSize(int value) {
    pixelBoxSize_ = value;
}

int Showpalette::getPixelBoxSize() const {
    return pixelBoxSize_;
}

std::string Showpalette::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "showpalette";

    bool first = true;

    if (pixelBoxSize_ != 30) {
        desc << (first ? "=" : ":") << "s=" << pixelBoxSize_;
        first = false;
    }

    return desc.str();
}
