#include "Tile.hpp"
#include <sstream>

Tile::Tile(std::pair<int, int> layout, int nb_frames, int margin, int padding, const std::string& color, int overlap, int init_padding) {
    // Initialize member variables from parameters
    this->layout_ = layout;
    this->nb_frames_ = nb_frames;
    this->margin_ = margin;
    this->padding_ = padding;
    this->color_ = color;
    this->overlap_ = overlap;
    this->init_padding_ = init_padding;
}

Tile::~Tile() {
    // Destructor implementation (if needed)
}

void Tile::setLayout(const std::pair<int, int>& value) {
    layout_ = value;
}

std::pair<int, int> Tile::getLayout() const {
    return layout_;
}

void Tile::setNb_frames(int value) {
    nb_frames_ = value;
}

int Tile::getNb_frames() const {
    return nb_frames_;
}

void Tile::setMargin(int value) {
    margin_ = value;
}

int Tile::getMargin() const {
    return margin_;
}

void Tile::setPadding(int value) {
    padding_ = value;
}

int Tile::getPadding() const {
    return padding_;
}

void Tile::setColor(const std::string& value) {
    color_ = value;
}

std::string Tile::getColor() const {
    return color_;
}

void Tile::setOverlap(int value) {
    overlap_ = value;
}

int Tile::getOverlap() const {
    return overlap_;
}

void Tile::setInit_padding(int value) {
    init_padding_ = value;
}

int Tile::getInit_padding() const {
    return init_padding_;
}

std::string Tile::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "tile";

    bool first = true;

    if (layout_.first != 0 || layout_.second != 1) {
        desc << (first ? "=" : ":") << "layout=" << layout_.first << "/" << layout_.second;
        first = false;
    }
    if (nb_frames_ != 0) {
        desc << (first ? "=" : ":") << "nb_frames=" << nb_frames_;
        first = false;
    }
    if (margin_ != 0) {
        desc << (first ? "=" : ":") << "margin=" << margin_;
        first = false;
    }
    if (padding_ != 0) {
        desc << (first ? "=" : ":") << "padding=" << padding_;
        first = false;
    }
    if (color_ != "black") {
        desc << (first ? "=" : ":") << "color=" << color_;
        first = false;
    }
    if (overlap_ != 0) {
        desc << (first ? "=" : ":") << "overlap=" << overlap_;
        first = false;
    }
    if (init_padding_ != 0) {
        desc << (first ? "=" : ":") << "init_padding=" << init_padding_;
        first = false;
    }

    return desc.str();
}
