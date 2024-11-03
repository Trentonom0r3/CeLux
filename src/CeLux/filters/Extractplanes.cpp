#include "Extractplanes.hpp"
#include <sstream>

Extractplanes::Extractplanes(int planes) {
    // Initialize member variables from parameters
    this->planes_ = planes;
}

Extractplanes::~Extractplanes() {
    // Destructor implementation (if needed)
}

void Extractplanes::setPlanes(int value) {
    planes_ = value;
}

int Extractplanes::getPlanes() const {
    return planes_;
}

std::string Extractplanes::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "extractplanes";

    bool first = true;

    if (planes_ != 1) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
