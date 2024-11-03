#include "Maskedmin.hpp"
#include <sstream>

Maskedmin::Maskedmin(int planes) {
    // Initialize member variables from parameters
    this->planes_ = planes;
}

Maskedmin::~Maskedmin() {
    // Destructor implementation (if needed)
}

void Maskedmin::setPlanes(int value) {
    planes_ = value;
}

int Maskedmin::getPlanes() const {
    return planes_;
}

std::string Maskedmin::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "maskedmin";

    bool first = true;

    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
