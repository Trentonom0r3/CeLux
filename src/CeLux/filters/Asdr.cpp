#include "Asdr.hpp"
#include <sstream>

Asdr::Asdr() {
    // Initialize member variables with default values
}

Asdr::~Asdr() {
    // Destructor implementation (if needed)
}

std::string Asdr::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "asdr";

    bool first = true;


    return desc.str();
}
