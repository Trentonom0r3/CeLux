#include "Asisdr.hpp"
#include <sstream>

Asisdr::Asisdr() {
    // Initialize member variables with default values
}

Asisdr::~Asisdr() {
    // Destructor implementation (if needed)
}

std::string Asisdr::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "asisdr";

    bool first = true;


    return desc.str();
}
