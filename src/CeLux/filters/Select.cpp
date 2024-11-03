#include "Select.hpp"
#include <sstream>

Select::Select(const std::string& expr, int outputs) {
    // Initialize member variables from parameters
    this->expr_ = expr;
    this->outputs_ = outputs;
}

Select::~Select() {
    // Destructor implementation (if needed)
}

void Select::setExpr(const std::string& value) {
    expr_ = value;
}

std::string Select::getExpr() const {
    return expr_;
}

void Select::setOutputs(int value) {
    outputs_ = value;
}

int Select::getOutputs() const {
    return outputs_;
}

std::string Select::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "select";

    bool first = true;

    if (expr_ != "1") {
        desc << (first ? "=" : ":") << "expr=" << expr_;
        first = false;
    }
    if (outputs_ != 1) {
        desc << (first ? "=" : ":") << "outputs=" << outputs_;
        first = false;
    }

    return desc.str();
}
