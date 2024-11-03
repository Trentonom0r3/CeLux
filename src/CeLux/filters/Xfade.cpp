#include "Xfade.hpp"
#include <sstream>

Xfade::Xfade(int transition, int64_t duration, int64_t offset, const std::string& expr) {
    // Initialize member variables from parameters
    this->transition_ = transition;
    this->duration_ = duration;
    this->offset_ = offset;
    this->expr_ = expr;
}

Xfade::~Xfade() {
    // Destructor implementation (if needed)
}

void Xfade::setTransition(int value) {
    transition_ = value;
}

int Xfade::getTransition() const {
    return transition_;
}

void Xfade::setDuration(int64_t value) {
    duration_ = value;
}

int64_t Xfade::getDuration() const {
    return duration_;
}

void Xfade::setOffset(int64_t value) {
    offset_ = value;
}

int64_t Xfade::getOffset() const {
    return offset_;
}

void Xfade::setExpr(const std::string& value) {
    expr_ = value;
}

std::string Xfade::getExpr() const {
    return expr_;
}

std::string Xfade::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "xfade";

    bool first = true;

    if (transition_ != 0) {
        desc << (first ? "=" : ":") << "transition=" << transition_;
        first = false;
    }
    if (duration_ != 1000000ULL) {
        desc << (first ? "=" : ":") << "duration=" << duration_;
        first = false;
    }
    if (offset_ != 0ULL) {
        desc << (first ? "=" : ":") << "offset=" << offset_;
        first = false;
    }
    if (!expr_.empty()) {
        desc << (first ? "=" : ":") << "expr=" << expr_;
        first = false;
    }

    return desc.str();
}
