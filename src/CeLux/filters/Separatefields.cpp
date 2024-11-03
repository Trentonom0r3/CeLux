#include "Separatefields.hpp"
#include <sstream>

Separatefields::Separatefields() {
    // Initialize member variables with default values
}

Separatefields::~Separatefields() {
    // Destructor implementation (if needed)
}

std::string Separatefields::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "separatefields";

    bool first = true;


    return desc.str();
}
