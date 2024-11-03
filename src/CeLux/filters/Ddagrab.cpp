#include "Ddagrab.hpp"
#include <sstream>

Ddagrab::Ddagrab(int output_idx, bool draw_mouse, std::pair<int, int> framerate, std::pair<int, int> video_size, int offset_x, int offset_y, int output_fmt, bool allow_fallback, bool force_fmt, bool dup_frames) {
    // Initialize member variables from parameters
    this->output_idx_ = output_idx;
    this->draw_mouse_ = draw_mouse;
    this->framerate_ = framerate;
    this->video_size_ = video_size;
    this->offset_x_ = offset_x;
    this->offset_y_ = offset_y;
    this->output_fmt_ = output_fmt;
    this->allow_fallback_ = allow_fallback;
    this->force_fmt_ = force_fmt;
    this->dup_frames_ = dup_frames;
}

Ddagrab::~Ddagrab() {
    // Destructor implementation (if needed)
}

void Ddagrab::setOutput_idx(int value) {
    output_idx_ = value;
}

int Ddagrab::getOutput_idx() const {
    return output_idx_;
}

void Ddagrab::setDraw_mouse(bool value) {
    draw_mouse_ = value;
}

bool Ddagrab::getDraw_mouse() const {
    return draw_mouse_;
}

void Ddagrab::setFramerate(const std::pair<int, int>& value) {
    framerate_ = value;
}

std::pair<int, int> Ddagrab::getFramerate() const {
    return framerate_;
}

void Ddagrab::setVideo_size(const std::pair<int, int>& value) {
    video_size_ = value;
}

std::pair<int, int> Ddagrab::getVideo_size() const {
    return video_size_;
}

void Ddagrab::setOffset_x(int value) {
    offset_x_ = value;
}

int Ddagrab::getOffset_x() const {
    return offset_x_;
}

void Ddagrab::setOffset_y(int value) {
    offset_y_ = value;
}

int Ddagrab::getOffset_y() const {
    return offset_y_;
}

void Ddagrab::setOutput_fmt(int value) {
    output_fmt_ = value;
}

int Ddagrab::getOutput_fmt() const {
    return output_fmt_;
}

void Ddagrab::setAllow_fallback(bool value) {
    allow_fallback_ = value;
}

bool Ddagrab::getAllow_fallback() const {
    return allow_fallback_;
}

void Ddagrab::setForce_fmt(bool value) {
    force_fmt_ = value;
}

bool Ddagrab::getForce_fmt() const {
    return force_fmt_;
}

void Ddagrab::setDup_frames(bool value) {
    dup_frames_ = value;
}

bool Ddagrab::getDup_frames() const {
    return dup_frames_;
}

std::string Ddagrab::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "ddagrab";

    bool first = true;

    if (output_idx_ != 0) {
        desc << (first ? "=" : ":") << "output_idx=" << output_idx_;
        first = false;
    }
    if (draw_mouse_ != true) {
        desc << (first ? "=" : ":") << "draw_mouse=" << (draw_mouse_ ? "1" : "0");
        first = false;
    }
    if (framerate_.first != 0 || framerate_.second != 1) {
        desc << (first ? "=" : ":") << "framerate=" << framerate_.first << "/" << framerate_.second;
        first = false;
    }
    if (video_size_.first != 0 || video_size_.second != 1) {
        desc << (first ? "=" : ":") << "video_size=" << video_size_.first << "/" << video_size_.second;
        first = false;
    }
    if (offset_x_ != 0) {
        desc << (first ? "=" : ":") << "offset_x=" << offset_x_;
        first = false;
    }
    if (offset_y_ != 0) {
        desc << (first ? "=" : ":") << "offset_y=" << offset_y_;
        first = false;
    }
    if (output_fmt_ != 87) {
        desc << (first ? "=" : ":") << "output_fmt=" << output_fmt_;
        first = false;
    }
    if (allow_fallback_ != false) {
        desc << (first ? "=" : ":") << "allow_fallback=" << (allow_fallback_ ? "1" : "0");
        first = false;
    }
    if (force_fmt_ != false) {
        desc << (first ? "=" : ":") << "force_fmt=" << (force_fmt_ ? "1" : "0");
        first = false;
    }
    if (dup_frames_ != true) {
        desc << (first ? "=" : ":") << "dup_frames=" << (dup_frames_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
