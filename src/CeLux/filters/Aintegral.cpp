#include "Aintegral.hpp"
#include <sstream>

Aintegral::Aintegral() {
    // Initialize member variables with default values
}

Aintegral::~Aintegral() {
    // Destructor implementation (if needed)
}

std::string Aintegral::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "aintegral";

    bool first = true;


    return desc.str();
}
