#include "Random.hpp"
#include <sstream>

Random::Random(int frames, int64_t seed) {
    // Initialize member variables from parameters
    this->frames_ = frames;
    this->seed_ = seed;
}

Random::~Random() {
    // Destructor implementation (if needed)
}

void Random::setFrames(int value) {
    frames_ = value;
}

int Random::getFrames() const {
    return frames_;
}

void Random::setSeed(int64_t value) {
    seed_ = value;
}

int64_t Random::getSeed() const {
    return seed_;
}

std::string Random::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "random";

    bool first = true;

    if (frames_ != 30) {
        desc << (first ? "=" : ":") << "frames=" << frames_;
        first = false;
    }
    if (seed_ != 0) {
        desc << (first ? "=" : ":") << "seed=" << seed_;
        first = false;
    }

    return desc.str();
}
