#include "Qp.hpp"
#include <sstream>

Qp::Qp(const std::string& qp) {
    // Initialize member variables from parameters
    this->qp_ = qp;
}

Qp::~Qp() {
    // Destructor implementation (if needed)
}

void Qp::setQp(const std::string& value) {
    qp_ = value;
}

std::string Qp::getQp() const {
    return qp_;
}

std::string Qp::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "qp";

    bool first = true;

    if (!qp_.empty()) {
        desc << (first ? "=" : ":") << "qp=" << qp_;
        first = false;
    }

    return desc.str();
}
