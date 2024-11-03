#include "Waveform.hpp"
#include <sstream>

Waveform::Waveform(int mode, float intensity, bool mirror, int display, int components, int envelope, int filter, int graticule, float opacity, int flags, int scale, float bgopacity, float tint0, float tint1, int fitmode, int input) {
    // Initialize member variables from parameters
    this->mode_ = mode;
    this->intensity_ = intensity;
    this->mirror_ = mirror;
    this->display_ = display;
    this->components_ = components;
    this->envelope_ = envelope;
    this->filter_ = filter;
    this->graticule_ = graticule;
    this->opacity_ = opacity;
    this->flags_ = flags;
    this->scale_ = scale;
    this->bgopacity_ = bgopacity;
    this->tint0_ = tint0;
    this->tint1_ = tint1;
    this->fitmode_ = fitmode;
    this->input_ = input;
}

Waveform::~Waveform() {
    // Destructor implementation (if needed)
}

void Waveform::setMode(int value) {
    mode_ = value;
}

int Waveform::getMode() const {
    return mode_;
}

void Waveform::setIntensity(float value) {
    intensity_ = value;
}

float Waveform::getIntensity() const {
    return intensity_;
}

void Waveform::setMirror(bool value) {
    mirror_ = value;
}

bool Waveform::getMirror() const {
    return mirror_;
}

void Waveform::setDisplay(int value) {
    display_ = value;
}

int Waveform::getDisplay() const {
    return display_;
}

void Waveform::setComponents(int value) {
    components_ = value;
}

int Waveform::getComponents() const {
    return components_;
}

void Waveform::setEnvelope(int value) {
    envelope_ = value;
}

int Waveform::getEnvelope() const {
    return envelope_;
}

void Waveform::setFilter(int value) {
    filter_ = value;
}

int Waveform::getFilter() const {
    return filter_;
}

void Waveform::setGraticule(int value) {
    graticule_ = value;
}

int Waveform::getGraticule() const {
    return graticule_;
}

void Waveform::setOpacity(float value) {
    opacity_ = value;
}

float Waveform::getOpacity() const {
    return opacity_;
}

void Waveform::setFlags(int value) {
    flags_ = value;
}

int Waveform::getFlags() const {
    return flags_;
}

void Waveform::setScale(int value) {
    scale_ = value;
}

int Waveform::getScale() const {
    return scale_;
}

void Waveform::setBgopacity(float value) {
    bgopacity_ = value;
}

float Waveform::getBgopacity() const {
    return bgopacity_;
}

void Waveform::setTint0(float value) {
    tint0_ = value;
}

float Waveform::getTint0() const {
    return tint0_;
}

void Waveform::setTint1(float value) {
    tint1_ = value;
}

float Waveform::getTint1() const {
    return tint1_;
}

void Waveform::setFitmode(int value) {
    fitmode_ = value;
}

int Waveform::getFitmode() const {
    return fitmode_;
}

void Waveform::setInput(int value) {
    input_ = value;
}

int Waveform::getInput() const {
    return input_;
}

std::string Waveform::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "waveform";

    bool first = true;

    if (mode_ != 1) {
        desc << (first ? "=" : ":") << "mode=" << mode_;
        first = false;
    }
    if (intensity_ != 0.04) {
        desc << (first ? "=" : ":") << "intensity=" << intensity_;
        first = false;
    }
    if (mirror_ != true) {
        desc << (first ? "=" : ":") << "mirror=" << (mirror_ ? "1" : "0");
        first = false;
    }
    if (display_ != 1) {
        desc << (first ? "=" : ":") << "display=" << display_;
        first = false;
    }
    if (components_ != 1) {
        desc << (first ? "=" : ":") << "components=" << components_;
        first = false;
    }
    if (envelope_ != 0) {
        desc << (first ? "=" : ":") << "envelope=" << envelope_;
        first = false;
    }
    if (filter_ != 0) {
        desc << (first ? "=" : ":") << "filter=" << filter_;
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
    if (flags_ != 1) {
        desc << (first ? "=" : ":") << "flags=" << flags_;
        first = false;
    }
    if (scale_ != 0) {
        desc << (first ? "=" : ":") << "scale=" << scale_;
        first = false;
    }
    if (bgopacity_ != 0.75) {
        desc << (first ? "=" : ":") << "bgopacity=" << bgopacity_;
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
    if (fitmode_ != 0) {
        desc << (first ? "=" : ":") << "fitmode=" << fitmode_;
        first = false;
    }
    if (input_ != 1) {
        desc << (first ? "=" : ":") << "input=" << input_;
        first = false;
    }

    return desc.str();
}
