#include "Apsnr.hpp"
#include <sstream>

Apsnr::Apsnr() {
    // Initialize member variables with default values
}

Apsnr::~Apsnr() {
    // Destructor implementation (if needed)
}

std::string Apsnr::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "apsnr";

    bool first = true;


    return desc.str();
}
