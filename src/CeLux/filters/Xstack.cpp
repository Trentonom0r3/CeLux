#include "Xstack.hpp"
#include <sstream>

Xstack::Xstack(int inputs, const std::string& layout, std::pair<int, int> grid, bool shortest, const std::string& fill) {
    // Initialize member variables from parameters
    this->inputs_ = inputs;
    this->layout_ = layout;
    this->grid_ = grid;
    this->shortest_ = shortest;
    this->fill_ = fill;
}

Xstack::~Xstack() {
    // Destructor implementation (if needed)
}

void Xstack::setInputs(int value) {
    inputs_ = value;
}

int Xstack::getInputs() const {
    return inputs_;
}

void Xstack::setLayout(const std::string& value) {
    layout_ = value;
}

std::string Xstack::getLayout() const {
    return layout_;
}

void Xstack::setGrid(const std::pair<int, int>& value) {
    grid_ = value;
}

std::pair<int, int> Xstack::getGrid() const {
    return grid_;
}

void Xstack::setShortest(bool value) {
    shortest_ = value;
}

bool Xstack::getShortest() const {
    return shortest_;
}

void Xstack::setFill(const std::string& value) {
    fill_ = value;
}

std::string Xstack::getFill() const {
    return fill_;
}

std::string Xstack::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "xstack";

    bool first = true;

    if (inputs_ != 2) {
        desc << (first ? "=" : ":") << "inputs=" << inputs_;
        first = false;
    }
    if (!layout_.empty()) {
        desc << (first ? "=" : ":") << "layout=" << layout_;
        first = false;
    }
    if (grid_.first != 0 || grid_.second != 1) {
        desc << (first ? "=" : ":") << "grid=" << grid_.first << "/" << grid_.second;
        first = false;
    }
    if (shortest_ != false) {
        desc << (first ? "=" : ":") << "shortest=" << (shortest_ ? "1" : "0");
        first = false;
    }
    if (fill_ != "none") {
        desc << (first ? "=" : ":") << "fill=" << fill_;
        first = false;
    }

    return desc.str();
}
