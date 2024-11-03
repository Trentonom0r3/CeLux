#include "Vflip.hpp"
#include <sstream>

Vflip::Vflip() {
    // Initialize member variables with default values
}

Vflip::~Vflip() {
    // Destructor implementation (if needed)
}

std::string Vflip::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "vflip";

    bool first = true;


    return desc.str();
}
