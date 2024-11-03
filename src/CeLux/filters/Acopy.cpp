#include "Acopy.hpp"
#include <sstream>

Acopy::Acopy() {
    // Initialize member variables with default values
}

Acopy::~Acopy() {
    // Destructor implementation (if needed)
}

std::string Acopy::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "acopy";

    bool first = true;


    return desc.str();
}
