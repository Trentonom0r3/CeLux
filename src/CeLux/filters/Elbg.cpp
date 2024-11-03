#include "Elbg.hpp"
#include <sstream>

Elbg::Elbg(int codebook_length, int nb_steps, int64_t seed, bool pal8, bool use_alpha) {
    // Initialize member variables from parameters
    this->codebook_length_ = codebook_length;
    this->nb_steps_ = nb_steps;
    this->seed_ = seed;
    this->pal8_ = pal8;
    this->use_alpha_ = use_alpha;
}

Elbg::~Elbg() {
    // Destructor implementation (if needed)
}

void Elbg::setCodebook_length(int value) {
    codebook_length_ = value;
}

int Elbg::getCodebook_length() const {
    return codebook_length_;
}

void Elbg::setNb_steps(int value) {
    nb_steps_ = value;
}

int Elbg::getNb_steps() const {
    return nb_steps_;
}

void Elbg::setSeed(int64_t value) {
    seed_ = value;
}

int64_t Elbg::getSeed() const {
    return seed_;
}

void Elbg::setPal8(bool value) {
    pal8_ = value;
}

bool Elbg::getPal8() const {
    return pal8_;
}

void Elbg::setUse_alpha(bool value) {
    use_alpha_ = value;
}

bool Elbg::getUse_alpha() const {
    return use_alpha_;
}

std::string Elbg::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "elbg";

    bool first = true;

    if (codebook_length_ != 256) {
        desc << (first ? "=" : ":") << "codebook_length=" << codebook_length_;
        first = false;
    }
    if (nb_steps_ != 1) {
        desc << (first ? "=" : ":") << "nb_steps=" << nb_steps_;
        first = false;
    }
    if (seed_ != 0) {
        desc << (first ? "=" : ":") << "seed=" << seed_;
        first = false;
    }
    if (pal8_ != false) {
        desc << (first ? "=" : ":") << "pal8=" << (pal8_ ? "1" : "0");
        first = false;
    }
    if (use_alpha_ != false) {
        desc << (first ? "=" : ":") << "use_alpha=" << (use_alpha_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
