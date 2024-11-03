#include "Corr.hpp"
#include <sstream>

Corr::Corr() {
    // Initialize member variables with default values
}

Corr::~Corr() {
    // Destructor implementation (if needed)
}

std::string Corr::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "corr";

    bool first = true;


    return desc.str();
}
