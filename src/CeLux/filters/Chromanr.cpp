#include "Chromanr.hpp"
#include <sstream>

Chromanr::Chromanr(float thres, int sizew, int sizeh, int stepw, int steph, float threy, float threu, float threv, int distance) {
    // Initialize member variables from parameters
    this->thres_ = thres;
    this->sizew_ = sizew;
    this->sizeh_ = sizeh;
    this->stepw_ = stepw;
    this->steph_ = steph;
    this->threy_ = threy;
    this->threu_ = threu;
    this->threv_ = threv;
    this->distance_ = distance;
}

Chromanr::~Chromanr() {
    // Destructor implementation (if needed)
}

void Chromanr::setThres(float value) {
    thres_ = value;
}

float Chromanr::getThres() const {
    return thres_;
}

void Chromanr::setSizew(int value) {
    sizew_ = value;
}

int Chromanr::getSizew() const {
    return sizew_;
}

void Chromanr::setSizeh(int value) {
    sizeh_ = value;
}

int Chromanr::getSizeh() const {
    return sizeh_;
}

void Chromanr::setStepw(int value) {
    stepw_ = value;
}

int Chromanr::getStepw() const {
    return stepw_;
}

void Chromanr::setSteph(int value) {
    steph_ = value;
}

int Chromanr::getSteph() const {
    return steph_;
}

void Chromanr::setThrey(float value) {
    threy_ = value;
}

float Chromanr::getThrey() const {
    return threy_;
}

void Chromanr::setThreu(float value) {
    threu_ = value;
}

float Chromanr::getThreu() const {
    return threu_;
}

void Chromanr::setThrev(float value) {
    threv_ = value;
}

float Chromanr::getThrev() const {
    return threv_;
}

void Chromanr::setDistance(int value) {
    distance_ = value;
}

int Chromanr::getDistance() const {
    return distance_;
}

std::string Chromanr::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "chromanr";

    bool first = true;

    if (thres_ != 30.00) {
        desc << (first ? "=" : ":") << "thres=" << thres_;
        first = false;
    }
    if (sizew_ != 5) {
        desc << (first ? "=" : ":") << "sizew=" << sizew_;
        first = false;
    }
    if (sizeh_ != 5) {
        desc << (first ? "=" : ":") << "sizeh=" << sizeh_;
        first = false;
    }
    if (stepw_ != 1) {
        desc << (first ? "=" : ":") << "stepw=" << stepw_;
        first = false;
    }
    if (steph_ != 1) {
        desc << (first ? "=" : ":") << "steph=" << steph_;
        first = false;
    }
    if (threy_ != 200.00) {
        desc << (first ? "=" : ":") << "threy=" << threy_;
        first = false;
    }
    if (threu_ != 200.00) {
        desc << (first ? "=" : ":") << "threu=" << threu_;
        first = false;
    }
    if (threv_ != 200.00) {
        desc << (first ? "=" : ":") << "threv=" << threv_;
        first = false;
    }
    if (distance_ != 0) {
        desc << (first ? "=" : ":") << "distance=" << distance_;
        first = false;
    }

    return desc.str();
}
