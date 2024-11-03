#include "Alatency.hpp"
#include <sstream>

Alatency::Alatency() {
    // Initialize member variables with default values
}

Alatency::~Alatency() {
    // Destructor implementation (if needed)
}

std::string Alatency::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "alatency";

    bool first = true;


    return desc.str();
}
