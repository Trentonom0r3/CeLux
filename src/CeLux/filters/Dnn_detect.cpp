#include "Dnn_detect.hpp"
#include <sstream>

Dnn_detect::Dnn_detect(int dnn_backend, const std::string& model, const std::string& input, const std::string& output, const std::string& backend_configs, bool async, float confidence, const std::string& labels, int model_type, int cell_w, int cell_h, int nb_classes, const std::string& anchors) {
    // Initialize member variables from parameters
    this->dnn_backend_ = dnn_backend;
    this->model_ = model;
    this->input_ = input;
    this->output_ = output;
    this->backend_configs_ = backend_configs;
    this->async_ = async;
    this->confidence_ = confidence;
    this->labels_ = labels;
    this->model_type_ = model_type;
    this->cell_w_ = cell_w;
    this->cell_h_ = cell_h;
    this->nb_classes_ = nb_classes;
    this->anchors_ = anchors;
}

Dnn_detect::~Dnn_detect() {
    // Destructor implementation (if needed)
}

void Dnn_detect::setDnn_backend(int value) {
    dnn_backend_ = value;
}

int Dnn_detect::getDnn_backend() const {
    return dnn_backend_;
}

void Dnn_detect::setModel(const std::string& value) {
    model_ = value;
}

std::string Dnn_detect::getModel() const {
    return model_;
}

void Dnn_detect::setInput(const std::string& value) {
    input_ = value;
}

std::string Dnn_detect::getInput() const {
    return input_;
}

void Dnn_detect::setOutput(const std::string& value) {
    output_ = value;
}

std::string Dnn_detect::getOutput() const {
    return output_;
}

void Dnn_detect::setBackend_configs(const std::string& value) {
    backend_configs_ = value;
}

std::string Dnn_detect::getBackend_configs() const {
    return backend_configs_;
}

void Dnn_detect::setAsync(bool value) {
    async_ = value;
}

bool Dnn_detect::getAsync() const {
    return async_;
}

void Dnn_detect::setConfidence(float value) {
    confidence_ = value;
}

float Dnn_detect::getConfidence() const {
    return confidence_;
}

void Dnn_detect::setLabels(const std::string& value) {
    labels_ = value;
}

std::string Dnn_detect::getLabels() const {
    return labels_;
}

void Dnn_detect::setModel_type(int value) {
    model_type_ = value;
}

int Dnn_detect::getModel_type() const {
    return model_type_;
}

void Dnn_detect::setCell_w(int value) {
    cell_w_ = value;
}

int Dnn_detect::getCell_w() const {
    return cell_w_;
}

void Dnn_detect::setCell_h(int value) {
    cell_h_ = value;
}

int Dnn_detect::getCell_h() const {
    return cell_h_;
}

void Dnn_detect::setNb_classes(int value) {
    nb_classes_ = value;
}

int Dnn_detect::getNb_classes() const {
    return nb_classes_;
}

void Dnn_detect::setAnchors(const std::string& value) {
    anchors_ = value;
}

std::string Dnn_detect::getAnchors() const {
    return anchors_;
}

std::string Dnn_detect::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "dnn_detect";

    bool first = true;

    if (dnn_backend_ != 2) {
        desc << (first ? "=" : ":") << "dnn_backend=" << dnn_backend_;
        first = false;
    }
    if (!model_.empty()) {
        desc << (first ? "=" : ":") << "model=" << model_;
        first = false;
    }
    if (!input_.empty()) {
        desc << (first ? "=" : ":") << "input=" << input_;
        first = false;
    }
    if (!output_.empty()) {
        desc << (first ? "=" : ":") << "output=" << output_;
        first = false;
    }
    if (!backend_configs_.empty()) {
        desc << (first ? "=" : ":") << "backend_configs=" << backend_configs_;
        first = false;
    }
    if (async_ != true) {
        desc << (first ? "=" : ":") << "async=" << (async_ ? "1" : "0");
        first = false;
    }
    if (confidence_ != 0.50) {
        desc << (first ? "=" : ":") << "confidence=" << confidence_;
        first = false;
    }
    if (!labels_.empty()) {
        desc << (first ? "=" : ":") << "labels=" << labels_;
        first = false;
    }
    if (model_type_ != 0) {
        desc << (first ? "=" : ":") << "model_type=" << model_type_;
        first = false;
    }
    if (cell_w_ != 0) {
        desc << (first ? "=" : ":") << "cell_w=" << cell_w_;
        first = false;
    }
    if (cell_h_ != 0) {
        desc << (first ? "=" : ":") << "cell_h=" << cell_h_;
        first = false;
    }
    if (nb_classes_ != 0) {
        desc << (first ? "=" : ":") << "nb_classes=" << nb_classes_;
        first = false;
    }
    if (!anchors_.empty()) {
        desc << (first ? "=" : ":") << "anchors=" << anchors_;
        first = false;
    }

    return desc.str();
}
