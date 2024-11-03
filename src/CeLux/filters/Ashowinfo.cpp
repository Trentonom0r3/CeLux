#include "Ashowinfo.hpp"
#include <sstream>

Ashowinfo::Ashowinfo() {
    // Initialize member variables with default values
}

Ashowinfo::~Ashowinfo() {
    // Destructor implementation (if needed)
}

std::string Ashowinfo::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "ashowinfo";

    bool first = true;


    return desc.str();
}
