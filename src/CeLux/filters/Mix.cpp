#include "Mix.hpp"
#include <sstream>

Mix::Mix(int inputs, const std::string& weights, float scale, int planes, int duration) {
    // Initialize member variables from parameters
    this->inputs_ = inputs;
    this->weights_ = weights;
    this->scale_ = scale;
    this->planes_ = planes;
    this->duration_ = duration;
}

Mix::~Mix() {
    // Destructor implementation (if needed)
}

void Mix::setInputs(int value) {
    inputs_ = value;
}

int Mix::getInputs() const {
    return inputs_;
}

void Mix::setWeights(const std::string& value) {
    weights_ = value;
}

std::string Mix::getWeights() const {
    return weights_;
}

void Mix::setScale(float value) {
    scale_ = value;
}

float Mix::getScale() const {
    return scale_;
}

void Mix::setPlanes(int value) {
    planes_ = value;
}

int Mix::getPlanes() const {
    return planes_;
}

void Mix::setDuration(int value) {
    duration_ = value;
}

int Mix::getDuration() const {
    return duration_;
}

std::string Mix::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "mix";

    bool first = true;

    if (inputs_ != 2) {
        desc << (first ? "=" : ":") << "inputs=" << inputs_;
        first = false;
    }
    if (weights_ != "1 1") {
        desc << (first ? "=" : ":") << "weights=" << weights_;
        first = false;
    }
    if (scale_ != 0.00) {
        desc << (first ? "=" : ":") << "scale=" << scale_;
        first = false;
    }
    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }
    if (duration_ != 0) {
        desc << (first ? "=" : ":") << "duration=" << duration_;
        first = false;
    }

    return desc.str();
}
