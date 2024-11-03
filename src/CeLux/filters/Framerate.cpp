#include "Framerate.hpp"
#include <sstream>

Framerate::Framerate(std::pair<int, int> fps, int interp_start, int interp_end, double scene, int flags) {
    // Initialize member variables from parameters
    this->fps_ = fps;
    this->interp_start_ = interp_start;
    this->interp_end_ = interp_end;
    this->scene_ = scene;
    this->flags_ = flags;
}

Framerate::~Framerate() {
    // Destructor implementation (if needed)
}

void Framerate::setFps(const std::pair<int, int>& value) {
    fps_ = value;
}

std::pair<int, int> Framerate::getFps() const {
    return fps_;
}

void Framerate::setInterp_start(int value) {
    interp_start_ = value;
}

int Framerate::getInterp_start() const {
    return interp_start_;
}

void Framerate::setInterp_end(int value) {
    interp_end_ = value;
}

int Framerate::getInterp_end() const {
    return interp_end_;
}

void Framerate::setScene(double value) {
    scene_ = value;
}

double Framerate::getScene() const {
    return scene_;
}

void Framerate::setFlags(int value) {
    flags_ = value;
}

int Framerate::getFlags() const {
    return flags_;
}

std::string Framerate::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "framerate";

    bool first = true;

    if (fps_.first != 0 || fps_.second != 1) {
        desc << (first ? "=" : ":") << "fps=" << fps_.first << "/" << fps_.second;
        first = false;
    }
    if (interp_start_ != 15) {
        desc << (first ? "=" : ":") << "interp_start=" << interp_start_;
        first = false;
    }
    if (interp_end_ != 240) {
        desc << (first ? "=" : ":") << "interp_end=" << interp_end_;
        first = false;
    }
    if (scene_ != 8.20) {
        desc << (first ? "=" : ":") << "scene=" << scene_;
        first = false;
    }
    if (flags_ != 1) {
        desc << (first ? "=" : ":") << "flags=" << flags_;
        first = false;
    }

    return desc.str();
}
