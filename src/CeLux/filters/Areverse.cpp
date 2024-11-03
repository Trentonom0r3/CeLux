#include "Areverse.hpp"
#include <sstream>

Areverse::Areverse() {
    // Initialize member variables with default values
}

Areverse::~Areverse() {
    // Destructor implementation (if needed)
}

std::string Areverse::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "areverse";

    bool first = true;


    return desc.str();
}
