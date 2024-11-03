#include "Null.hpp"
#include <sstream>

Null::Null() {
    // Initialize member variables with default values
}

Null::~Null() {
    // Destructor implementation (if needed)
}

std::string Null::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "null";

    bool first = true;


    return desc.str();
}
