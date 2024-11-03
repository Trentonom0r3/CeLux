#include "Tpad.hpp"
#include <sstream>

Tpad::Tpad(int start, int stop, int start_mode, int stop_mode, int64_t start_duration, int64_t stop_duration, const std::string& color) {
    // Initialize member variables from parameters
    this->start_ = start;
    this->stop_ = stop;
    this->start_mode_ = start_mode;
    this->stop_mode_ = stop_mode;
    this->start_duration_ = start_duration;
    this->stop_duration_ = stop_duration;
    this->color_ = color;
}

Tpad::~Tpad() {
    // Destructor implementation (if needed)
}

void Tpad::setStart(int value) {
    start_ = value;
}

int Tpad::getStart() const {
    return start_;
}

void Tpad::setStop(int value) {
    stop_ = value;
}

int Tpad::getStop() const {
    return stop_;
}

void Tpad::setStart_mode(int value) {
    start_mode_ = value;
}

int Tpad::getStart_mode() const {
    return start_mode_;
}

void Tpad::setStop_mode(int value) {
    stop_mode_ = value;
}

int Tpad::getStop_mode() const {
    return stop_mode_;
}

void Tpad::setStart_duration(int64_t value) {
    start_duration_ = value;
}

int64_t Tpad::getStart_duration() const {
    return start_duration_;
}

void Tpad::setStop_duration(int64_t value) {
    stop_duration_ = value;
}

int64_t Tpad::getStop_duration() const {
    return stop_duration_;
}

void Tpad::setColor(const std::string& value) {
    color_ = value;
}

std::string Tpad::getColor() const {
    return color_;
}

std::string Tpad::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "tpad";

    bool first = true;

    if (start_ != 0) {
        desc << (first ? "=" : ":") << "start=" << start_;
        first = false;
    }
    if (stop_ != 0) {
        desc << (first ? "=" : ":") << "stop=" << stop_;
        first = false;
    }
    if (start_mode_ != 0) {
        desc << (first ? "=" : ":") << "start_mode=" << start_mode_;
        first = false;
    }
    if (stop_mode_ != 0) {
        desc << (first ? "=" : ":") << "stop_mode=" << stop_mode_;
        first = false;
    }
    if (start_duration_ != 0ULL) {
        desc << (first ? "=" : ":") << "start_duration=" << start_duration_;
        first = false;
    }
    if (stop_duration_ != 0ULL) {
        desc << (first ? "=" : ":") << "stop_duration=" << stop_duration_;
        first = false;
    }
    if (color_ != "black") {
        desc << (first ? "=" : ":") << "color=" << color_;
        first = false;
    }

    return desc.str();
}
