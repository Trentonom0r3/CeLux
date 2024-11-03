#include "Setpts.hpp"
#include <sstream>

Setpts::Setpts(const std::string& expr) {
    // Initialize member variables from parameters
    this->expr_ = expr;
}

Setpts::~Setpts() {
    // Destructor implementation (if needed)
}

void Setpts::setExpr(const std::string& value) {
    expr_ = value;
}

std::string Setpts::getExpr() const {
    return expr_;
}

std::string Setpts::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "setpts";

    bool first = true;

    if (expr_ != "PTS") {
        desc << (first ? "=" : ":") << "expr=" << expr_;
        first = false;
    }

    return desc.str();
}
