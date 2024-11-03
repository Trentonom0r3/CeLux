#include "Aderivative.hpp"
#include <sstream>

Aderivative::Aderivative() {
    // Initialize member variables with default values
}

Aderivative::~Aderivative() {
    // Destructor implementation (if needed)
}

std::string Aderivative::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "aderivative";

    bool first = true;


    return desc.str();
}
