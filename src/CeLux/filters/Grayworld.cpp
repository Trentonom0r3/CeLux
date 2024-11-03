#include "Grayworld.hpp"
#include <sstream>

Grayworld::Grayworld() {
    // Initialize member variables with default values
}

Grayworld::~Grayworld() {
    // Destructor implementation (if needed)
}

std::string Grayworld::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "grayworld";

    bool first = true;


    return desc.str();
}
