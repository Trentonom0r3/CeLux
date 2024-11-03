#include "Vmafmotion.hpp"
#include <sstream>

Vmafmotion::Vmafmotion(const std::string& stats_file) {
    // Initialize member variables from parameters
    this->stats_file_ = stats_file;
}

Vmafmotion::~Vmafmotion() {
    // Destructor implementation (if needed)
}

void Vmafmotion::setStats_file(const std::string& value) {
    stats_file_ = value;
}

std::string Vmafmotion::getStats_file() const {
    return stats_file_;
}

std::string Vmafmotion::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "vmafmotion";

    bool first = true;

    if (!stats_file_.empty()) {
        desc << (first ? "=" : ":") << "stats_file=" << stats_file_;
        first = false;
    }

    return desc.str();
}
