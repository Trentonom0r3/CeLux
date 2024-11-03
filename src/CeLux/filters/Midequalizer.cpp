#include "Midequalizer.hpp"
#include <sstream>

Midequalizer::Midequalizer(int planes) {
    // Initialize member variables from parameters
    this->planes_ = planes;
}

Midequalizer::~Midequalizer() {
    // Destructor implementation (if needed)
}

void Midequalizer::setPlanes(int value) {
    planes_ = value;
}

int Midequalizer::getPlanes() const {
    return planes_;
}

std::string Midequalizer::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "midequalizer";

    bool first = true;

    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }

    return desc.str();
}
