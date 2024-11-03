#include "Reverse.hpp"
#include <sstream>

Reverse::Reverse() {
    // Initialize member variables with default values
}

Reverse::~Reverse() {
    // Destructor implementation (if needed)
}

std::string Reverse::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "reverse";

    bool first = true;


    return desc.str();
}
