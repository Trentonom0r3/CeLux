#include "Maskedmerge.hpp"
#include <sstream>

Maskedmerge::Maskedmerge(int planes) {
    // Initialize member variables from parameters
    this->planes_ = planes;
}

Maskedmerge::~Maskedmerge() {
    // Destructor implementation (if needed)
}

void Maskedmerge::setPlanes(int value) {
    planes_ = value;
}

int Maskedmerge::getPlanes() const {
    return planes_;
}

std::string Maskedmerge::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "maskedmerge";

    bool first = true;

    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
