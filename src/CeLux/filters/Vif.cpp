#include "Vif.hpp"
#include <sstream>

Vif::Vif() {
    // Initialize member variables with default values
}

Vif::~Vif() {
    // Destructor implementation (if needed)
}

std::string Vif::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "vif";

    bool first = true;


    return desc.str();
}
