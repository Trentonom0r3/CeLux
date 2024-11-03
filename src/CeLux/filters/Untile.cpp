#include "Untile.hpp"
#include <sstream>

Untile::Untile(std::pair<int, int> layout) {
    // Initialize member variables from parameters
    this->layout_ = layout;
}

Untile::~Untile() {
    // Destructor implementation (if needed)
}

void Untile::setLayout(const std::pair<int, int>& value) {
    layout_ = value;
}

std::pair<int, int> Untile::getLayout() const {
    return layout_;
}

std::string Untile::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "untile";

    bool first = true;

    if (layout_.first != 0 || layout_.second != 1) {
        desc << (first ? "=" : ":") << "layout=" << layout_.first << "/" << layout_.second;
        first = false;
    }

    return desc.str();
}
