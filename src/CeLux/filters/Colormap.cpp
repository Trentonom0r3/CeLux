#include "Colormap.hpp"
#include <sstream>

Colormap::Colormap(std::pair<int, int> patch_size, int nb_patches, int type, int kernel) {
    // Initialize member variables from parameters
    this->patch_size_ = patch_size;
    this->nb_patches_ = nb_patches;
    this->type_ = type;
    this->kernel_ = kernel;
}

Colormap::~Colormap() {
    // Destructor implementation (if needed)
}

void Colormap::setPatch_size(const std::pair<int, int>& value) {
    patch_size_ = value;
}

std::pair<int, int> Colormap::getPatch_size() const {
    return patch_size_;
}

void Colormap::setNb_patches(int value) {
    nb_patches_ = value;
}

int Colormap::getNb_patches() const {
    return nb_patches_;
}

void Colormap::setType(int value) {
    type_ = value;
}

int Colormap::getType() const {
    return type_;
}

void Colormap::setKernel(int value) {
    kernel_ = value;
}

int Colormap::getKernel() const {
    return kernel_;
}

std::string Colormap::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "colormap";

    bool first = true;

    if (patch_size_.first != 0 || patch_size_.second != 1) {
        desc << (first ? "=" : ":") << "patch_size=" << patch_size_.first << "/" << patch_size_.second;
        first = false;
    }
    if (nb_patches_ != 0) {
        desc << (first ? "=" : ":") << "nb_patches=" << nb_patches_;
        first = false;
    }
    if (type_ != 1) {
        desc << (first ? "=" : ":") << "type=" << type_;
        first = false;
    }
    if (kernel_ != 0) {
        desc << (first ? "=" : ":") << "kernel=" << kernel_;
        first = false;
    }

    return desc.str();
}
