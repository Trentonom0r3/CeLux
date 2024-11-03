#include "Alphaextract.hpp"
#include <sstream>

Alphaextract::Alphaextract() {
    // Initialize member variables with default values
}

Alphaextract::~Alphaextract() {
    // Destructor implementation (if needed)
}

std::string Alphaextract::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "alphaextract";

    bool first = true;


    return desc.str();
}
