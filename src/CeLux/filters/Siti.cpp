#include "Siti.hpp"
#include <sstream>

Siti::Siti(bool print_summary) {
    // Initialize member variables from parameters
    this->print_summary_ = print_summary;
}

Siti::~Siti() {
    // Destructor implementation (if needed)
}

void Siti::setPrint_summary(bool value) {
    print_summary_ = value;
}

bool Siti::getPrint_summary() const {
    return print_summary_;
}

std::string Siti::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "siti";

    bool first = true;

    if (print_summary_ != false) {
        desc << (first ? "=" : ":") << "print_summary=" << (print_summary_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
