#include "Ssim360.hpp"
#include <sstream>

Ssim360::Ssim360(const std::string& stats_file, int compute_chroma, int frame_skip_ratio, int ref_projection, int main_projection, int ref_stereo, int main_stereo, float ref_pad, float main_pad, int use_tape, const std::string& heatmap_str, int default_heatmap_width, int default_heatmap_height) {
    // Initialize member variables from parameters
    this->stats_file_ = stats_file;
    this->compute_chroma_ = compute_chroma;
    this->frame_skip_ratio_ = frame_skip_ratio;
    this->ref_projection_ = ref_projection;
    this->main_projection_ = main_projection;
    this->ref_stereo_ = ref_stereo;
    this->main_stereo_ = main_stereo;
    this->ref_pad_ = ref_pad;
    this->main_pad_ = main_pad;
    this->use_tape_ = use_tape;
    this->heatmap_str_ = heatmap_str;
    this->default_heatmap_width_ = default_heatmap_width;
    this->default_heatmap_height_ = default_heatmap_height;
}

Ssim360::~Ssim360() {
    // Destructor implementation (if needed)
}

void Ssim360::setStats_file(const std::string& value) {
    stats_file_ = value;
}

std::string Ssim360::getStats_file() const {
    return stats_file_;
}

void Ssim360::setCompute_chroma(int value) {
    compute_chroma_ = value;
}

int Ssim360::getCompute_chroma() const {
    return compute_chroma_;
}

void Ssim360::setFrame_skip_ratio(int value) {
    frame_skip_ratio_ = value;
}

int Ssim360::getFrame_skip_ratio() const {
    return frame_skip_ratio_;
}

void Ssim360::setRef_projection(int value) {
    ref_projection_ = value;
}

int Ssim360::getRef_projection() const {
    return ref_projection_;
}

void Ssim360::setMain_projection(int value) {
    main_projection_ = value;
}

int Ssim360::getMain_projection() const {
    return main_projection_;
}

void Ssim360::setRef_stereo(int value) {
    ref_stereo_ = value;
}

int Ssim360::getRef_stereo() const {
    return ref_stereo_;
}

void Ssim360::setMain_stereo(int value) {
    main_stereo_ = value;
}

int Ssim360::getMain_stereo() const {
    return main_stereo_;
}

void Ssim360::setRef_pad(float value) {
    ref_pad_ = value;
}

float Ssim360::getRef_pad() const {
    return ref_pad_;
}

void Ssim360::setMain_pad(float value) {
    main_pad_ = value;
}

float Ssim360::getMain_pad() const {
    return main_pad_;
}

void Ssim360::setUse_tape(int value) {
    use_tape_ = value;
}

int Ssim360::getUse_tape() const {
    return use_tape_;
}

void Ssim360::setHeatmap_str(const std::string& value) {
    heatmap_str_ = value;
}

std::string Ssim360::getHeatmap_str() const {
    return heatmap_str_;
}

void Ssim360::setDefault_heatmap_width(int value) {
    default_heatmap_width_ = value;
}

int Ssim360::getDefault_heatmap_width() const {
    return default_heatmap_width_;
}

void Ssim360::setDefault_heatmap_height(int value) {
    default_heatmap_height_ = value;
}

int Ssim360::getDefault_heatmap_height() const {
    return default_heatmap_height_;
}

std::string Ssim360::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "ssim360";

    bool first = true;

    if (!stats_file_.empty()) {
        desc << (first ? "=" : ":") << "stats_file=" << stats_file_;
        first = false;
    }
    if (compute_chroma_ != 1) {
        desc << (first ? "=" : ":") << "compute_chroma=" << compute_chroma_;
        first = false;
    }
    if (frame_skip_ratio_ != 0) {
        desc << (first ? "=" : ":") << "frame_skip_ratio=" << frame_skip_ratio_;
        first = false;
    }
    if (ref_projection_ != 4) {
        desc << (first ? "=" : ":") << "ref_projection=" << ref_projection_;
        first = false;
    }
    if (main_projection_ != 5) {
        desc << (first ? "=" : ":") << "main_projection=" << main_projection_;
        first = false;
    }
    if (ref_stereo_ != 2) {
        desc << (first ? "=" : ":") << "ref_stereo=" << ref_stereo_;
        first = false;
    }
    if (main_stereo_ != 3) {
        desc << (first ? "=" : ":") << "main_stereo=" << main_stereo_;
        first = false;
    }
    if (ref_pad_ != 0.00) {
        desc << (first ? "=" : ":") << "ref_pad=" << ref_pad_;
        first = false;
    }
    if (main_pad_ != 0.00) {
        desc << (first ? "=" : ":") << "main_pad=" << main_pad_;
        first = false;
    }
    if (use_tape_ != 0) {
        desc << (first ? "=" : ":") << "use_tape=" << use_tape_;
        first = false;
    }
    if (!heatmap_str_.empty()) {
        desc << (first ? "=" : ":") << "heatmap_str=" << heatmap_str_;
        first = false;
    }
    if (default_heatmap_width_ != 32) {
        desc << (first ? "=" : ":") << "default_heatmap_width=" << default_heatmap_width_;
        first = false;
    }
    if (default_heatmap_height_ != 16) {
        desc << (first ? "=" : ":") << "default_heatmap_height=" << default_heatmap_height_;
        first = false;
    }

    return desc.str();
}
