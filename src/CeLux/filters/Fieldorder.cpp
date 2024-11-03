#include "Fieldorder.hpp"
#include <sstream>

Fieldorder::Fieldorder(int order) {
    // Initialize member variables from parameters
    this->order_ = order;
}

Fieldorder::~Fieldorder() {
    // Destructor implementation (if needed)
}

void Fieldorder::setOrder(int value) {
    order_ = value;
}

int Fieldorder::getOrder() const {
    return order_;
}

std::string Fieldorder::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "fieldorder";

    bool first = true;

    if (order_ != 1) {
        desc << (first ? "=" : ":") << "order=" << order_;
        first = false;
    }

    return desc.str();
}
