#include "Hflip.hpp"
#include <sstream>

Hflip::Hflip() {
    // Initialize member variables with default values
}

Hflip::~Hflip() {
    // Destructor implementation (if needed)
}

std::string Hflip::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "hflip";

    bool first = true;


    return desc.str();
}
