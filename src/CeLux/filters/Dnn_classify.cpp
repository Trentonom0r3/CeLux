#include "Dnn_classify.hpp"
#include <sstream>

Dnn_classify::Dnn_classify(int dnn_backend, const std::string& model, const std::string& input, const std::string& output, const std::string& backend_configs, bool async, float confidence, const std::string& labels, const std::string& target) {
    // Initialize member variables from parameters
    this->dnn_backend_ = dnn_backend;
    this->model_ = model;
    this->input_ = input;
    this->output_ = output;
    this->backend_configs_ = backend_configs;
    this->async_ = async;
    this->confidence_ = confidence;
    this->labels_ = labels;
    this->target_ = target;
}

Dnn_classify::~Dnn_classify() {
    // Destructor implementation (if needed)
}

void Dnn_classify::setDnn_backend(int value) {
    dnn_backend_ = value;
}

int Dnn_classify::getDnn_backend() const {
    return dnn_backend_;
}

void Dnn_classify::setModel(const std::string& value) {
    model_ = value;
}

std::string Dnn_classify::getModel() const {
    return model_;
}

void Dnn_classify::setInput(const std::string& value) {
    input_ = value;
}

std::string Dnn_classify::getInput() const {
    return input_;
}

void Dnn_classify::setOutput(const std::string& value) {
    output_ = value;
}

std::string Dnn_classify::getOutput() const {
    return output_;
}

void Dnn_classify::setBackend_configs(const std::string& value) {
    backend_configs_ = value;
}

std::string Dnn_classify::getBackend_configs() const {
    return backend_configs_;
}

void Dnn_classify::setAsync(bool value) {
    async_ = value;
}

bool Dnn_classify::getAsync() const {
    return async_;
}

void Dnn_classify::setConfidence(float value) {
    confidence_ = value;
}

float Dnn_classify::getConfidence() const {
    return confidence_;
}

void Dnn_classify::setLabels(const std::string& value) {
    labels_ = value;
}

std::string Dnn_classify::getLabels() const {
    return labels_;
}

void Dnn_classify::setTarget(const std::string& value) {
    target_ = value;
}

std::string Dnn_classify::getTarget() const {
    return target_;
}

std::string Dnn_classify::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "dnn_classify";

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
    if (!target_.empty()) {
        desc << (first ? "=" : ":") << "target=" << target_;
        first = false;
    }

    return desc.str();
}
