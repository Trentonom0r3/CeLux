#include "Ccrepack.hpp"
#include <sstream>

Ccrepack::Ccrepack() {
    // Initialize member variables with default values
}

Ccrepack::~Ccrepack() {
    // Destructor implementation (if needed)
}

std::string Ccrepack::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "ccrepack";

    bool first = true;


    return desc.str();
}
