#include "Threshold.hpp"
#include <sstream>

Threshold::Threshold(int planes) {
    // Initialize member variables from parameters
    this->planes_ = planes;
}

Threshold::~Threshold() {
    // Destructor implementation (if needed)
}

void Threshold::setPlanes(int value) {
    planes_ = value;
}

int Threshold::getPlanes() const {
    return planes_;
}

std::string Threshold::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "threshold";

    bool first = true;

    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
