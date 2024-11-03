#include "Ssim.hpp"
#include <sstream>

Ssim::Ssim(const std::string& stats_file) {
    // Initialize member variables from parameters
    this->stats_file_ = stats_file;
}

Ssim::~Ssim() {
    // Destructor implementation (if needed)
}

void Ssim::setStats_file(const std::string& value) {
    stats_file_ = value;
}

std::string Ssim::getStats_file() const {
    return stats_file_;
}

std::string Ssim::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "ssim";

    bool first = true;

    if (!stats_file_.empty()) {
        desc << (first ? "=" : ":") << "stats_file=" << stats_file_;
        first = false;
    }

    return desc.str();
}
