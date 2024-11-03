#include "Palettegen.hpp"
#include <sstream>

Palettegen::Palettegen(int max_colors, bool reserve_transparent, const std::string& transparency_color, int stats_mode) {
    // Initialize member variables from parameters
    this->max_colors_ = max_colors;
    this->reserve_transparent_ = reserve_transparent;
    this->transparency_color_ = transparency_color;
    this->stats_mode_ = stats_mode;
}

Palettegen::~Palettegen() {
    // Destructor implementation (if needed)
}

void Palettegen::setMax_colors(int value) {
    max_colors_ = value;
}

int Palettegen::getMax_colors() const {
    return max_colors_;
}

void Palettegen::setReserve_transparent(bool value) {
    reserve_transparent_ = value;
}

bool Palettegen::getReserve_transparent() const {
    return reserve_transparent_;
}

void Palettegen::setTransparency_color(const std::string& value) {
    transparency_color_ = value;
}

std::string Palettegen::getTransparency_color() const {
    return transparency_color_;
}

void Palettegen::setStats_mode(int value) {
    stats_mode_ = value;
}

int Palettegen::getStats_mode() const {
    return stats_mode_;
}

std::string Palettegen::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "palettegen";

    bool first = true;

    if (max_colors_ != 256) {
        desc << (first ? "=" : ":") << "max_colors=" << max_colors_;
        first = false;
    }
    if (reserve_transparent_ != true) {
        desc << (first ? "=" : ":") << "reserve_transparent=" << (reserve_transparent_ ? "1" : "0");
        first = false;
    }
    if (transparency_color_ != "lime") {
        desc << (first ? "=" : ":") << "transparency_color=" << transparency_color_;
        first = false;
    }
    if (stats_mode_ != 0) {
        desc << (first ? "=" : ":") << "stats_mode=" << stats_mode_;
        first = false;
    }

    return desc.str();
}
