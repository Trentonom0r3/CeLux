#include "Vfrdet.hpp"
#include <sstream>

Vfrdet::Vfrdet() {
    // Initialize member variables with default values
}

Vfrdet::~Vfrdet() {
    // Destructor implementation (if needed)
}

std::string Vfrdet::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "vfrdet";

    bool first = true;


    return desc.str();
}
