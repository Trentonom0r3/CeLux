#include "Copy.hpp"
#include <sstream>

Copy::Copy() {
    // Initialize member variables with default values
}

Copy::~Copy() {
    // Destructor implementation (if needed)
}

std::string Copy::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "copy";

    bool first = true;


    return desc.str();
}
