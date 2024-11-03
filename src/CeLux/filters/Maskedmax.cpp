#include "Maskedmax.hpp"
#include <sstream>

Maskedmax::Maskedmax(int planes) {
    // Initialize member variables from parameters
    this->planes_ = planes;
}

Maskedmax::~Maskedmax() {
    // Destructor implementation (if needed)
}

void Maskedmax::setPlanes(int value) {
    planes_ = value;
}

int Maskedmax::getPlanes() const {
    return planes_;
}

std::string Maskedmax::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "maskedmax";

    bool first = true;

    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
