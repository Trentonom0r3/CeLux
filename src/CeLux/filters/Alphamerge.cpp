#include "Alphamerge.hpp"
#include <sstream>

Alphamerge::Alphamerge() {
    // Initialize member variables with default values
}

Alphamerge::~Alphamerge() {
    // Destructor implementation (if needed)
}

std::string Alphamerge::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "alphamerge";

    bool first = true;


    return desc.str();
}
