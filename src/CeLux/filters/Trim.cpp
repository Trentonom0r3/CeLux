#include "Trim.hpp"
#include <sstream>

Trim::Trim(int64_t starti, int64_t endi, int64_t start_pts, int64_t end_pts, int64_t durationi, int64_t start_frame, int64_t end_frame) {
    // Initialize member variables from parameters
    this->starti_ = starti;
    this->endi_ = endi;
    this->start_pts_ = start_pts;
    this->end_pts_ = end_pts;
    this->durationi_ = durationi;
    this->start_frame_ = start_frame;
    this->end_frame_ = end_frame;
}

Trim::~Trim() {
    // Destructor implementation (if needed)
}

void Trim::setStarti(int64_t value) {
    starti_ = value;
}

int64_t Trim::getStarti() const {
    return starti_;
}

void Trim::setEndi(int64_t value) {
    endi_ = value;
}

int64_t Trim::getEndi() const {
    return endi_;
}

void Trim::setStart_pts(int64_t value) {
    start_pts_ = value;
}

int64_t Trim::getStart_pts() const {
    return start_pts_;
}

void Trim::setEnd_pts(int64_t value) {
    end_pts_ = value;
}

int64_t Trim::getEnd_pts() const {
    return end_pts_;
}

void Trim::setDurationi(int64_t value) {
    durationi_ = value;
}

int64_t Trim::getDurationi() const {
    return durationi_;
}

void Trim::setStart_frame(int64_t value) {
    start_frame_ = value;
}

int64_t Trim::getStart_frame() const {
    return start_frame_;
}

void Trim::setEnd_frame(int64_t value) {
    end_frame_ = value;
}

int64_t Trim::getEnd_frame() const {
    return end_frame_;
}

std::string Trim::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "trim";

    bool first = true;

    if (starti_ != 9223372036854775807ULL) {
        desc << (first ? "=" : ":") << "starti=" << starti_;
        first = false;
    }
    if (endi_ != 9223372036854775807ULL) {
        desc << (first ? "=" : ":") << "endi=" << endi_;
        first = false;
    }
    if (start_pts_ != 0) {
        desc << (first ? "=" : ":") << "start_pts=" << start_pts_;
        first = false;
    }
    if (end_pts_ != 0) {
        desc << (first ? "=" : ":") << "end_pts=" << end_pts_;
        first = false;
    }
    if (durationi_ != 0ULL) {
        desc << (first ? "=" : ":") << "durationi=" << durationi_;
        first = false;
    }
    if (start_frame_ != 0) {
        desc << (first ? "=" : ":") << "start_frame=" << start_frame_;
        first = false;
    }
    if (end_frame_ != 9223372036854775807ULL) {
        desc << (first ? "=" : ":") << "end_frame=" << end_frame_;
        first = false;
    }

    return desc.str();
}
