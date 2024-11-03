#include "Loop.hpp"
#include <sstream>

Loop::Loop(int loop, int64_t size, int64_t start, int64_t time) {
    // Initialize member variables from parameters
    this->loop_ = loop;
    this->size_ = size;
    this->start_ = start;
    this->time_ = time;
}

Loop::~Loop() {
    // Destructor implementation (if needed)
}

void Loop::setLoop(int value) {
    loop_ = value;
}

int Loop::getLoop() const {
    return loop_;
}

void Loop::setSize(int64_t value) {
    size_ = value;
}

int64_t Loop::getSize() const {
    return size_;
}

void Loop::setStart(int64_t value) {
    start_ = value;
}

int64_t Loop::getStart() const {
    return start_;
}

void Loop::setTime(int64_t value) {
    time_ = value;
}

int64_t Loop::getTime() const {
    return time_;
}

std::string Loop::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "loop";

    bool first = true;

    if (loop_ != 0) {
        desc << (first ? "=" : ":") << "loop=" << loop_;
        first = false;
    }
    if (size_ != 0ULL) {
        desc << (first ? "=" : ":") << "size=" << size_;
        first = false;
    }
    if (start_ != 0ULL) {
        desc << (first ? "=" : ":") << "start=" << start_;
        first = false;
    }
    if (time_ != 9223372036854775807ULL) {
        desc << (first ? "=" : ":") << "time=" << time_;
        first = false;
    }

    return desc.str();
}
