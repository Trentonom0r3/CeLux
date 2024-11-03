#include "Swapuv.hpp"
#include <sstream>

Swapuv::Swapuv() {
    // Initialize member variables with default values
}

Swapuv::~Swapuv() {
    // Destructor implementation (if needed)
}

std::string Swapuv::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "swapuv";

    bool first = true;


    return desc.str();
}
