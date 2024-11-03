#include "Buffersink.hpp"
#include <sstream>

Buffersink::Buffersink(std::vector<std::string> pix_fmts, std::vector<uint8_t> color_spaces, std::vector<uint8_t> color_ranges) {
    // Initialize member variables from parameters
    this->pix_fmts_ = pix_fmts;
    this->color_spaces_ = color_spaces;
    this->color_ranges_ = color_ranges;
}

Buffersink::~Buffersink() {
    // Destructor implementation (if needed)
}

void Buffersink::setPix_fmts(const std::vector<std::string>& value) {
    pix_fmts_ = value;
}

std::vector<std::string> Buffersink::getPix_fmts() const {
    return pix_fmts_;
}

void Buffersink::setColor_spaces(const std::vector<uint8_t>& value) {
    color_spaces_ = value;
}

std::vector<uint8_t> Buffersink::getColor_spaces() const {
    return color_spaces_;
}

void Buffersink::setColor_ranges(const std::vector<uint8_t>& value) {
    color_ranges_ = value;
}

std::vector<uint8_t> Buffersink::getColor_ranges() const {
    return color_ranges_;
}

std::string Buffersink::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "buffersink";

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
        desc << (first ? "=" : ":") << "color_spaces=";
        for (size_t i = 0; i < color_spaces_.size(); ++i) {
            desc << static_cast<int>(color_spaces_[i]);
            if (i != color_spaces_.size() - 1) desc << ",";
        }
        first = false;
    }
    if (!color_ranges_.empty()) {
        desc << (first ? "=" : ":") << "color_ranges=";
        for (size_t i = 0; i < color_ranges_.size(); ++i) {
            desc << static_cast<int>(color_ranges_[i]);
            if (i != color_ranges_.size() - 1) desc << ",";
        }
        first = false;
    }

    return desc.str();
}
