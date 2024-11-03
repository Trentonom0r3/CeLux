#include "Vectorscope.hpp"
#include <sstream>

Vectorscope::Vectorscope(int mode, int colorComponentOnXAxis, int colorComponentOnYAxis, float intensity, int envelope, int graticule, float opacity, int flags, float bgopacity, float lthreshold, float hthreshold, int colorspace, float tint0, float tint1) {
    // Initialize member variables from parameters
    this->mode_ = mode;
    this->colorComponentOnXAxis_ = colorComponentOnXAxis;
    this->colorComponentOnYAxis_ = colorComponentOnYAxis;
    this->intensity_ = intensity;
    this->envelope_ = envelope;
    this->graticule_ = graticule;
    this->opacity_ = opacity;
    this->flags_ = flags;
    this->bgopacity_ = bgopacity;
    this->lthreshold_ = lthreshold;
    this->hthreshold_ = hthreshold;
    this->colorspace_ = colorspace;
    this->tint0_ = tint0;
    this->tint1_ = tint1;
}

Vectorscope::~Vectorscope() {
    // Destructor implementation (if needed)
}

void Vectorscope::setMode(int value) {
    mode_ = value;
}

int Vectorscope::getMode() const {
    return mode_;
}

void Vectorscope::setColorComponentOnXAxis(int value) {
    colorComponentOnXAxis_ = value;
}

int Vectorscope::getColorComponentOnXAxis() const {
    return colorComponentOnXAxis_;
}

void Vectorscope::setColorComponentOnYAxis(int value) {
    colorComponentOnYAxis_ = value;
}

int Vectorscope::getColorComponentOnYAxis() const {
    return colorComponentOnYAxis_;
}

void Vectorscope::setIntensity(float value) {
    intensity_ = value;
}

float Vectorscope::getIntensity() const {
    return intensity_;
}

void Vectorscope::setEnvelope(int value) {
    envelope_ = value;
}

int Vectorscope::getEnvelope() const {
    return envelope_;
}

void Vectorscope::setGraticule(int value) {
    graticule_ = value;
}

int Vectorscope::getGraticule() const {
    return graticule_;
}

void Vectorscope::setOpacity(float value) {
    opacity_ = value;
}

float Vectorscope::getOpacity() const {
    return opacity_;
}

void Vectorscope::setFlags(int value) {
    flags_ = value;
}

int Vectorscope::getFlags() const {
    return flags_;
}

void Vectorscope::setBgopacity(float value) {
    bgopacity_ = value;
}

float Vectorscope::getBgopacity() const {
    return bgopacity_;
}

void Vectorscope::setLthreshold(float value) {
    lthreshold_ = value;
}

float Vectorscope::getLthreshold() const {
    return lthreshold_;
}

void Vectorscope::setHthreshold(float value) {
    hthreshold_ = value;
}

float Vectorscope::getHthreshold() const {
    return hthreshold_;
}

void Vectorscope::setColorspace(int value) {
    colorspace_ = value;
}

int Vectorscope::getColorspace() const {
    return colorspace_;
}

void Vectorscope::setTint0(float value) {
    tint0_ = value;
}

float Vectorscope::getTint0() const {
    return tint0_;
}

void Vectorscope::setTint1(float value) {
    tint1_ = value;
}

float Vectorscope::getTint1() const {
    return tint1_;
}

std::string Vectorscope::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "vectorscope";

    bool first = true;

    if (mode_ != 0) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (colorComponentOnXAxis_ != 1) {
        desc << (first ? "=" : ":") << "x=" << colorComponentOnXAxis_;
        first = false;
    }
    if (colorComponentOnYAxis_ != 2) {
        desc << (first ? "=" : ":") << "y=" << colorComponentOnYAxis_;
        first = false;
    }
    if (intensity_ != 0.00) {
        desc << (first ? "=" : ":") << "intensity=" << intensity_;
        first = false;
    }
    if (envelope_ != 0) {
        desc << (first ? "=" : ":") << "envelope=" << envelope_;
        first = false;
    }
    if (graticule_ != 0) {
        desc << (first ? "=" : ":") << "graticule=" << graticule_;
        first = false;
    }
    if (opacity_ != 0.75) {
        desc << (first ? "=" : ":") << "opacity=" << opacity_;
        first = false;
    }
    if (flags_ != 4) {
        desc << (first ? "=" : ":") << "flags=" << flags_;
        first = false;
    }
    if (bgopacity_ != 0.30) {
        desc << (first ? "=" : ":") << "bgopacity=" << bgopacity_;
        first = false;
    }
    if (lthreshold_ != 0.00) {
        desc << (first ? "=" : ":") << "lthreshold=" << lthreshold_;
        first = false;
    }
    if (hthreshold_ != 1.00) {
        desc << (first ? "=" : ":") << "hthreshold=" << hthreshold_;
        first = false;
    }
    if (colorspace_ != 0) {
        desc << (first ? "=" : ":") << "colorspace=" << colorspace_;
        first = false;
    }
    if (tint0_ != 0.00) {
        desc << (first ? "=" : ":") << "tint0=" << tint0_;
        first = false;
    }
    if (tint1_ != 0.00) {
        desc << (first ? "=" : ":") << "tint1=" << tint1_;
        first = false;
    }

    return desc.str();
}
