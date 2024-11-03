#include "Dnn_processing.hpp"
#include <sstream>

Dnn_processing::Dnn_processing(int dnn_backend, const std::string& model, const std::string& input, const std::string& output, const std::string& backend_configs, bool async) {
    // Initialize member variables from parameters
    this->dnn_backend_ = dnn_backend;
    this->model_ = model;
    this->input_ = input;
    this->output_ = output;
    this->backend_configs_ = backend_configs;
    this->async_ = async;
}

Dnn_processing::~Dnn_processing() {
    // Destructor implementation (if needed)
}

void Dnn_processing::setDnn_backend(int value) {
    dnn_backend_ = value;
}

int Dnn_processing::getDnn_backend() const {
    return dnn_backend_;
}

void Dnn_processing::setModel(const std::string& value) {
    model_ = value;
}

std::string Dnn_processing::getModel() const {
    return model_;
}

void Dnn_processing::setInput(const std::string& value) {
    input_ = value;
}

std::string Dnn_processing::getInput() const {
    return input_;
}

void Dnn_processing::setOutput(const std::string& value) {
    output_ = value;
}

std::string Dnn_processing::getOutput() const {
    return output_;
}

void Dnn_processing::setBackend_configs(const std::string& value) {
    backend_configs_ = value;
}

std::string Dnn_processing::getBackend_configs() const {
    return backend_configs_;
}

void Dnn_processing::setAsync(bool value) {
    async_ = value;
}

bool Dnn_processing::getAsync() const {
    return async_;
}

std::string Dnn_processing::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "dnn_processing";

    bool first = true;

    if (dnn_backend_ != 1) {
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

    return desc.str();
}
