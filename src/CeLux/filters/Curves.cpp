#include "Curves.hpp"
#include <sstream>

Curves::Curves(int preset, const std::string& master, const std::string& red, const std::string& green, const std::string& blue, const std::string& all, const std::string& psfile, const std::string& plot, int interp) {
    // Initialize member variables from parameters
    this->preset_ = preset;
    this->master_ = master;
    this->red_ = red;
    this->green_ = green;
    this->blue_ = blue;
    this->all_ = all;
    this->psfile_ = psfile;
    this->plot_ = plot;
    this->interp_ = interp;
}

Curves::~Curves() {
    // Destructor implementation (if needed)
}

void Curves::setPreset(int value) {
    preset_ = value;
}

int Curves::getPreset() const {
    return preset_;
}

void Curves::setMaster(const std::string& value) {
    master_ = value;
}

std::string Curves::getMaster() const {
    return master_;
}

void Curves::setRed(const std::string& value) {
    red_ = value;
}

std::string Curves::getRed() const {
    return red_;
}

void Curves::setGreen(const std::string& value) {
    green_ = value;
}

std::string Curves::getGreen() const {
    return green_;
}

void Curves::setBlue(const std::string& value) {
    blue_ = value;
}

std::string Curves::getBlue() const {
    return blue_;
}

void Curves::setAll(const std::string& value) {
    all_ = value;
}

std::string Curves::getAll() const {
    return all_;
}

void Curves::setPsfile(const std::string& value) {
    psfile_ = value;
}

std::string Curves::getPsfile() const {
    return psfile_;
}

void Curves::setPlot(const std::string& value) {
    plot_ = value;
}

std::string Curves::getPlot() const {
    return plot_;
}

void Curves::setInterp(int value) {
    interp_ = value;
}

int Curves::getInterp() const {
    return interp_;
}

std::string Curves::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "curves";

    bool first = true;

    if (preset_ != 0) {
        desc << (first ? "=" : ":") << "preset=" << preset_;
        first = false;
    }
    if (!master_.empty()) {
        desc << (first ? "=" : ":") << "master=" << master_;
        first = false;
    }
    if (!red_.empty()) {
        desc << (first ? "=" : ":") << "red=" << red_;
        first = false;
    }
    if (!green_.empty()) {
        desc << (first ? "=" : ":") << "green=" << green_;
        first = false;
    }
    if (!blue_.empty()) {
        desc << (first ? "=" : ":") << "blue=" << blue_;
        first = false;
    }
    if (!all_.empty()) {
        desc << (first ? "=" : ":") << "all=" << all_;
        first = false;
    }
    if (!psfile_.empty()) {
        desc << (first ? "=" : ":") << "psfile=" << psfile_;
        first = false;
    }
    if (!plot_.empty()) {
        desc << (first ? "=" : ":") << "plot=" << plot_;
        first = false;
    }
    if (interp_ != 0) {
        desc << (first ? "=" : ":") << "interp=" << interp_;
        first = false;
    }

    return desc.str();
}
