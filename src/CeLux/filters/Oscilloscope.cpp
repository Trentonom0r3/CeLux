#include "Oscilloscope.hpp"
#include <sstream>

Oscilloscope::Oscilloscope(float scopeXPosition, float scopeYPosition, float scopeSize, float scopeTilt, float traceOpacity, float tx, float ty, float tw, float th, int componentsToTrace, bool drawTraceGrid, bool st, bool sc) {
    // Initialize member variables from parameters
    this->scopeXPosition_ = scopeXPosition;
    this->scopeYPosition_ = scopeYPosition;
    this->scopeSize_ = scopeSize;
    this->scopeTilt_ = scopeTilt;
    this->traceOpacity_ = traceOpacity;
    this->tx_ = tx;
    this->ty_ = ty;
    this->tw_ = tw;
    this->th_ = th;
    this->componentsToTrace_ = componentsToTrace;
    this->drawTraceGrid_ = drawTraceGrid;
    this->st_ = st;
    this->sc_ = sc;
}

Oscilloscope::~Oscilloscope() {
    // Destructor implementation (if needed)
}

void Oscilloscope::setScopeXPosition(float value) {
    scopeXPosition_ = value;
}

float Oscilloscope::getScopeXPosition() const {
    return scopeXPosition_;
}

void Oscilloscope::setScopeYPosition(float value) {
    scopeYPosition_ = value;
}

float Oscilloscope::getScopeYPosition() const {
    return scopeYPosition_;
}

void Oscilloscope::setScopeSize(float value) {
    scopeSize_ = value;
}

float Oscilloscope::getScopeSize() const {
    return scopeSize_;
}

void Oscilloscope::setScopeTilt(float value) {
    scopeTilt_ = value;
}

float Oscilloscope::getScopeTilt() const {
    return scopeTilt_;
}

void Oscilloscope::setTraceOpacity(float value) {
    traceOpacity_ = value;
}

float Oscilloscope::getTraceOpacity() const {
    return traceOpacity_;
}

void Oscilloscope::setTx(float value) {
    tx_ = value;
}

float Oscilloscope::getTx() const {
    return tx_;
}

void Oscilloscope::setTy(float value) {
    ty_ = value;
}

float Oscilloscope::getTy() const {
    return ty_;
}

void Oscilloscope::setTw(float value) {
    tw_ = value;
}

float Oscilloscope::getTw() const {
    return tw_;
}

void Oscilloscope::setTh(float value) {
    th_ = value;
}

float Oscilloscope::getTh() const {
    return th_;
}

void Oscilloscope::setComponentsToTrace(int value) {
    componentsToTrace_ = value;
}

int Oscilloscope::getComponentsToTrace() const {
    return componentsToTrace_;
}

void Oscilloscope::setDrawTraceGrid(bool value) {
    drawTraceGrid_ = value;
}

bool Oscilloscope::getDrawTraceGrid() const {
    return drawTraceGrid_;
}

void Oscilloscope::setSt(bool value) {
    st_ = value;
}

bool Oscilloscope::getSt() const {
    return st_;
}

void Oscilloscope::setSc(bool value) {
    sc_ = value;
}

bool Oscilloscope::getSc() const {
    return sc_;
}

std::string Oscilloscope::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "oscilloscope";

    bool first = true;

    if (scopeXPosition_ != 0.50) {
        desc << (first ? "=" : ":") << "x=" << scopeXPosition_;
        first = false;
    }
    if (scopeYPosition_ != 0.50) {
        desc << (first ? "=" : ":") << "y=" << scopeYPosition_;
        first = false;
    }
    if (scopeSize_ != 0.80) {
        desc << (first ? "=" : ":") << "s=" << scopeSize_;
        first = false;
    }
    if (scopeTilt_ != 0.50) {
        desc << (first ? "=" : ":") << "t=" << scopeTilt_;
        first = false;
    }
    if (traceOpacity_ != 0.80) {
        desc << (first ? "=" : ":") << "o=" << traceOpacity_;
        first = false;
    }
    if (tx_ != 0.50) {
        desc << (first ? "=" : ":") << "tx=" << tx_;
        first = false;
    }
    if (ty_ != 0.90) {
        desc << (first ? "=" : ":") << "ty=" << ty_;
        first = false;
    }
    if (tw_ != 0.80) {
        desc << (first ? "=" : ":") << "tw=" << tw_;
        first = false;
    }
    if (th_ != 0.30) {
        desc << (first ? "=" : ":") << "th=" << th_;
        first = false;
    }
    if (componentsToTrace_ != 7) {
        desc << (first ? "=" : ":") << "c=" << componentsToTrace_;
        first = false;
    }
    if (drawTraceGrid_ != true) {
        desc << (first ? "=" : ":") << "g=" << (drawTraceGrid_ ? "1" : "0");
        first = false;
    }
    if (st_ != true) {
        desc << (first ? "=" : ":") << "st=" << (st_ ? "1" : "0");
        first = false;
    }
    if (sc_ != true) {
        desc << (first ? "=" : ":") << "sc=" << (sc_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
