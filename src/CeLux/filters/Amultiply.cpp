#include "Amultiply.hpp"
#include <sstream>

Amultiply::Amultiply() {
    // Initialize member variables with default values
}

Amultiply::~Amultiply() {
    // Destructor implementation (if needed)
}

std::string Amultiply::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "amultiply";

    bool first = true;


    return desc.str();
}
