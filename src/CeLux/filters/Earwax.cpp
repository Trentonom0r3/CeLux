#include "Earwax.hpp"
#include <sstream>

Earwax::Earwax() {
    // Initialize member variables with default values
}

Earwax::~Earwax() {
    // Destructor implementation (if needed)
}

std::string Earwax::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "earwax";

    bool first = true;


    return desc.str();
}
