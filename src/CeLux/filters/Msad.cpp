#include "Msad.hpp"
#include <sstream>

Msad::Msad() {
    // Initialize member variables with default values
}

Msad::~Msad() {
    // Destructor implementation (if needed)
}

std::string Msad::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "msad";

    bool first = true;


    return desc.str();
}
