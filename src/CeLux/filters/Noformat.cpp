#include "Noformat.hpp"
#include <sstream>

Noformat::Noformat(std::vector<std::string> pix_fmts, const std::string& color_spaces, const std::string& color_ranges) {
    // Initialize member variables from parameters
    this->pix_fmts_ = pix_fmts;
    this->color_spaces_ = color_spaces;
    this->color_ranges_ = color_ranges;
}

Noformat::~Noformat() {
    // Destructor implementation (if needed)
}

void Noformat::setPix_fmts(const std::vector<std::string>& value) {
    pix_fmts_ = value;
}

std::vector<std::string> Noformat::getPix_fmts() const {
    return pix_fmts_;
}

void Noformat::setColor_spaces(const std::string& value) {
    color_spaces_ = value;
}

std::string Noformat::getColor_spaces() const {
    return color_spaces_;
}

void Noformat::setColor_ranges(const std::string& value) {
    color_ranges_ = value;
}

std::string Noformat::getColor_ranges() const {
    return color_ranges_;
}

std::string Noformat::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "noformat";

    bool first = true;

    if (!pix_fmts_.empty()) {
        desc << (first ? "=" : ":") << "pix_fmts=";
        for (size_t i = 0; i < pix_fmts_.size(); ++i) {
            desc << pix_fmts_[i];
            if (i != pix_fmts_.size() - 1) desc << ",";
        }
        first = false;
    }
    if (!color_spaces_.empty()) {
        desc << (first ? "=" : ":") << "color_spaces=" << color_spaces_;
        first = false;
    }
    if (!color_ranges_.empty()) {
        desc << (first ? "=" : ":") << "color_ranges=" << color_ranges_;
        first = false;
    }

    return desc.str();
}
