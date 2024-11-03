#include "Thumbnail.hpp"
#include <sstream>

Thumbnail::Thumbnail(int framesBatchSize, int log) {
    // Initialize member variables from parameters
    this->framesBatchSize_ = framesBatchSize;
    this->log_ = log;
}

Thumbnail::~Thumbnail() {
    // Destructor implementation (if needed)
}

void Thumbnail::setFramesBatchSize(int value) {
    framesBatchSize_ = value;
}

int Thumbnail::getFramesBatchSize() const {
    return framesBatchSize_;
}

void Thumbnail::setLog(int value) {
    log_ = value;
}

int Thumbnail::getLog() const {
    return log_;
}

std::string Thumbnail::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "thumbnail";

    bool first = true;

    if (framesBatchSize_ != 100) {
        desc << (first ? "=" : ":") << "n=" << framesBatchSize_;
        first = false;
    }
    if (log_ != 32) {
        desc << (first ? "=" : ":") << "log=" << log_;
        first = false;
    }

    return desc.str();
}
