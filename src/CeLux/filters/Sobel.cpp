#include "Sobel.hpp"
#include <sstream>

Sobel::Sobel(int planes, float scale, float delta) {
    // Initialize member variables from parameters
    this->planes_ = planes;
    this->scale_ = scale;
    this->delta_ = delta;
}

Sobel::~Sobel() {
    // Destructor implementation (if needed)
}

void Sobel::setPlanes(int value) {
    planes_ = value;
}

int Sobel::getPlanes() const {
    return planes_;
}

void Sobel::setScale(float value) {
    scale_ = value;
}

float Sobel::getScale() const {
    return scale_;
}

void Sobel::setDelta(float value) {
    delta_ = value;
}

float Sobel::getDelta() const {
    return delta_;
}

std::string Sobel::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "sobel";

    bool first = true;

    if (planes_ != 15) {
        desc << (first ? "=" : ":") << "planes=" << planes_;
        first = false;
    }
    if (scale_ != 1.00) {
        desc << (first ? "=" : ":") << "scale=" << scale_;
        first = false;
    }
    if (delta_ != 0.00) {
        desc << (first ? "=" : ":") << "delta=" << delta_;
        first = false;
    }

    return desc.str();
}
