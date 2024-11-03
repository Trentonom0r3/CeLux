#include "Field.hpp"
#include <sstream>

Field::Field(int type) {
    // Initialize member variables from parameters
    this->type_ = type;
}

Field::~Field() {
    // Destructor implementation (if needed)
}

void Field::setType(int value) {
    type_ = value;
}

int Field::getType() const {
    return type_;
}

std::string Field::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "field";

    bool first = true;

    if (type_ != 0) {
        desc << (first ? "=" : ":") << "type=" << type_;
        first = false;
    }

    return desc.str();
}
