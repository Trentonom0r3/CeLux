#include "Displace.hpp"
#include <sstream>

Displace::Displace(int edge) {
    // Initialize member variables from parameters
    this->edge_ = edge;
}

Displace::~Displace() {
    // Destructor implementation (if needed)
}

void Displace::setEdge(int value) {
    edge_ = value;
}

int Displace::getEdge() const {
    return edge_;
}

std::string Displace::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "displace";

    bool first = true;

    if (edge_ != 1) {
        desc << (first ? "=" : ":") << "edge=" << edge_;
        first = false;
    }

    return desc.str();
}
