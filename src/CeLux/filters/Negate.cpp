#include "Negate.hpp"
#include <sstream>

Negate::Negate(int components, bool negate_alpha) {
    // Initialize member variables from parameters
    this->components_ = components;
    this->negate_alpha_ = negate_alpha;
}

Negate::~Negate() {
    // Destructor implementation (if needed)
}

void Negate::setComponents(int value) {
    components_ = value;
}

int Negate::getComponents() const {
    return components_;
}

void Negate::setNegate_alpha(bool value) {
    negate_alpha_ = value;
}

bool Negate::getNegate_alpha() const {
    return negate_alpha_;
}

std::string Negate::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "negate";

    bool first = true;

    if (components_ != 119) {
        desc << (first ? "=" : ":") << "components=" << components_;
        first = false;
    }
    if (negate_alpha_ != false) {
        desc << (first ? "=" : ":") << "negate_alpha=" << (negate_alpha_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
