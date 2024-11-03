#include "Volumedetect.hpp"
#include <sstream>

Volumedetect::Volumedetect() {
    // Initialize member variables with default values
}

Volumedetect::~Volumedetect() {
    // Destructor implementation (if needed)
}

std::string Volumedetect::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "volumedetect";

    bool first = true;


    return desc.str();
}
