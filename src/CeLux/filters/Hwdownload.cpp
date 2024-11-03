#include "Hwdownload.hpp"
#include <sstream>

Hwdownload::Hwdownload() {
    // Initialize member variables with default values
}

Hwdownload::~Hwdownload() {
    // Destructor implementation (if needed)
}

std::string Hwdownload::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "hwdownload";

    bool first = true;


    return desc.str();
}
