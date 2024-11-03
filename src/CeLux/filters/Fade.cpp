#include "Fade.hpp"
#include <sstream>

Fade::Fade(int type, int start_frame, int nb_frames, bool alpha, int64_t start_time, int64_t duration, const std::string& color) {
    // Initialize member variables from parameters
    this->type_ = type;
    this->start_frame_ = start_frame;
    this->nb_frames_ = nb_frames;
    this->alpha_ = alpha;
    this->start_time_ = start_time;
    this->duration_ = duration;
    this->color_ = color;
}

Fade::~Fade() {
    // Destructor implementation (if needed)
}

void Fade::setType(int value) {
    type_ = value;
}

int Fade::getType() const {
    return type_;
}

void Fade::setStart_frame(int value) {
    start_frame_ = value;
}

int Fade::getStart_frame() const {
    return start_frame_;
}

void Fade::setNb_frames(int value) {
    nb_frames_ = value;
}

int Fade::getNb_frames() const {
    return nb_frames_;
}

void Fade::setAlpha(bool value) {
    alpha_ = value;
}

bool Fade::getAlpha() const {
    return alpha_;
}

void Fade::setStart_time(int64_t value) {
    start_time_ = value;
}

int64_t Fade::getStart_time() const {
    return start_time_;
}

void Fade::setDuration(int64_t value) {
    duration_ = value;
}

int64_t Fade::getDuration() const {
    return duration_;
}

void Fade::setColor(const std::string& value) {
    color_ = value;
}

std::string Fade::getColor() const {
    return color_;
}

std::string Fade::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "fade";

    bool first = true;

    if (type_ != 0) {
        desc << (first ? "=" : ":") << "type=" << type_;
        first = false;
    }
    if (start_frame_ != 0) {
        desc << (first ? "=" : ":") << "start_frame=" << start_frame_;
        first = false;
    }
    if (nb_frames_ != 25) {
        desc << (first ? "=" : ":") << "nb_frames=" << nb_frames_;
        first = false;
    }
    if (alpha_ != false) {
        desc << (first ? "=" : ":") << "alpha=" << (alpha_ ? "1" : "0");
        first = false;
    }
    if (start_time_ != 0ULL) {
        desc << (first ? "=" : ":") << "start_time=" << start_time_;
        first = false;
    }
    if (duration_ != 0ULL) {
        desc << (first ? "=" : ":") << "duration=" << duration_;
        first = false;
    }
    if (color_ != "black") {
        desc << (first ? "=" : ":") << "color=" << color_;
        first = false;
    }

    return desc.str();
}
