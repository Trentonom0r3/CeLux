#include "Anull.hpp"
#include <sstream>

Anull::Anull() {
    // Initialize member variables with default values
}

Anull::~Anull() {
    // Destructor implementation (if needed)
}

std::string Anull::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "anull";

    bool first = true;


    return desc.str();
}
