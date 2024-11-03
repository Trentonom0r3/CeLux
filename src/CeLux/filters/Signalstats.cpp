#include "Signalstats.hpp"
#include <sstream>

Signalstats::Signalstats(int stat, int out, const std::string& color) {
    // Initialize member variables from parameters
    this->stat_ = stat;
    this->out_ = out;
    this->color_ = color;
}

Signalstats::~Signalstats() {
    // Destructor implementation (if needed)
}

void Signalstats::setStat(int value) {
    stat_ = value;
}

int Signalstats::getStat() const {
    return stat_;
}

void Signalstats::setOut(int value) {
    out_ = value;
}

int Signalstats::getOut() const {
    return out_;
}

void Signalstats::setColor(const std::string& value) {
    color_ = value;
}

std::string Signalstats::getColor() const {
    return color_;
}

std::string Signalstats::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "signalstats";

    bool first = true;

    if (stat_ != 0) {
        desc << (first ? "=" : ":") << "stat=" << stat_;
        first = false;
    }
    if (out_ != -1) {
        desc << (first ? "=" : ":") << "out=" << out_;
        first = false;
    }
    if (color_ != "yellow") {
        desc << (first ? "=" : ":") << "color=" << color_;
        first = false;
    }

    return desc.str();
}
