#include "Psnr.hpp"
#include <sstream>

Psnr::Psnr(const std::string& stats_file, int stats_version, bool output_max) {
    // Initialize member variables from parameters
    this->stats_file_ = stats_file;
    this->stats_version_ = stats_version;
    this->output_max_ = output_max;
}

Psnr::~Psnr() {
    // Destructor implementation (if needed)
}

void Psnr::setStats_file(const std::string& value) {
    stats_file_ = value;
}

std::string Psnr::getStats_file() const {
    return stats_file_;
}

void Psnr::setStats_version(int value) {
    stats_version_ = value;
}

int Psnr::getStats_version() const {
    return stats_version_;
}

void Psnr::setOutput_max(bool value) {
    output_max_ = value;
}

bool Psnr::getOutput_max() const {
    return output_max_;
}

std::string Psnr::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "psnr";

    bool first = true;

    if (!stats_file_.empty()) {
        desc << (first ? "=" : ":") << "stats_file=" << stats_file_;
        first = false;
    }
    if (stats_version_ != 1) {
        desc << (first ? "=" : ":") << "stats_version=" << stats_version_;
        first = false;
    }
    if (output_max_ != false) {
        desc << (first ? "=" : ":") << "output_max=" << (output_max_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
