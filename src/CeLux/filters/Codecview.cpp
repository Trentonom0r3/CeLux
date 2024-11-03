#include "Codecview.hpp"
#include <sstream>

Codecview::Codecview(int mv, bool qp, int mv_type, int frame_type, bool block) {
    // Initialize member variables from parameters
    this->mv_ = mv;
    this->qp_ = qp;
    this->mv_type_ = mv_type;
    this->frame_type_ = frame_type;
    this->block_ = block;
}

Codecview::~Codecview() {
    // Destructor implementation (if needed)
}

void Codecview::setMv(int value) {
    mv_ = value;
}

int Codecview::getMv() const {
    return mv_;
}

void Codecview::setQp(bool value) {
    qp_ = value;
}

bool Codecview::getQp() const {
    return qp_;
}

void Codecview::setMv_type(int value) {
    mv_type_ = value;
}

int Codecview::getMv_type() const {
    return mv_type_;
}

void Codecview::setFrame_type(int value) {
    frame_type_ = value;
}

int Codecview::getFrame_type() const {
    return frame_type_;
}

void Codecview::setBlock(bool value) {
    block_ = value;
}

bool Codecview::getBlock() const {
    return block_;
}

std::string Codecview::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "codecview";

    bool first = true;

    if (mv_ != 0) {
        desc << (first ? "=" : ":") << "mv=" << mv_;
        first = false;
    }
    if (qp_ != false) {
        desc << (first ? "=" : ":") << "qp=" << (qp_ ? "1" : "0");
        first = false;
    }
    if (mv_type_ != 0) {
        desc << (first ? "=" : ":") << "mv_type=" << mv_type_;
        first = false;
    }
    if (frame_type_ != 0) {
        desc << (first ? "=" : ":") << "frame_type=" << frame_type_;
        first = false;
    }
    if (block_ != false) {
        desc << (first ? "=" : ":") << "block=" << (block_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
