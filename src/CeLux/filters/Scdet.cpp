#include "Scdet.hpp"
#include <sstream>

Scdet::Scdet(double threshold, bool sc_pass) {
    // Initialize member variables from parameters
    this->threshold_ = threshold;
    this->sc_pass_ = sc_pass;
}

Scdet::~Scdet() {
    // Destructor implementation (if needed)
}

void Scdet::setThreshold(double value) {
    threshold_ = value;
}

double Scdet::getThreshold() const {
    return threshold_;
}

void Scdet::setSc_pass(bool value) {
    sc_pass_ = value;
}

bool Scdet::getSc_pass() const {
    return sc_pass_;
}

std::string Scdet::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "scdet";

    bool first = true;

    if (threshold_ != 10.00) {
        desc << (first ? "=" : ":") << "threshold=" << threshold_;
        first = false;
    }
    if (sc_pass_ != false) {
        desc << (first ? "=" : ":") << "sc_pass=" << (sc_pass_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
