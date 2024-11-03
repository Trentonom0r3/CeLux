#include "Settb.hpp"
#include <sstream>

Settb::Settb(const std::string& expr) {
    // Initialize member variables from parameters
    this->expr_ = expr;
}

Settb::~Settb() {
    // Destructor implementation (if needed)
}

void Settb::setExpr(const std::string& value) {
    expr_ = value;
}

std::string Settb::getExpr() const {
    return expr_;
}

std::string Settb::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "settb";

    bool first = true;

    if (expr_ != "intb") {
        desc << (first ? "=" : ":") << "expr=" << expr_;
        first = false;
    }

    return desc.str();
}
