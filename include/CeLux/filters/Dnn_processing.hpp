#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Dnn_processing : public FilterBase {
public:
    /**
     * Apply DNN processing filter to the input.
     */
    /**
     * DNN backend
     * Unit: backend
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setDnn_backend(int value);
    int getDnn_backend() const;

    /**
     * path to model file
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setModel(const std::string& value);
    std::string getModel() const;

    /**
     * input name of the model
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setInput(const std::string& value);
    std::string getInput() const;

    /**
     * output name of the model
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setOutput(const std::string& value);
    std::string getOutput() const;

    /**
     * backend configs
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setBackend_configs(const std::string& value);
    std::string getBackend_configs() const;

    /**
     * use DNN async inference (ignored, use backend_configs='async=1')
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setAsync(bool value);
    bool getAsync() const;

    Dnn_processing(int dnn_backend = 1, const std::string& model = "", const std::string& input = "", const std::string& output = "", const std::string& backend_configs = "", bool async = true);
    virtual ~Dnn_processing();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int dnn_backend_;
    std::string model_;
    std::string input_;
    std::string output_;
    std::string backend_configs_;
    bool async_;
};
