#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Dnn_classify : public FilterBase {
public:
    /**
     * Apply DNN classify filter to the input.
     */
    /**
     * DNN backend
     * Unit: backend
     * Type: Integer
     * Required: No
     * Default: 2
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

    /**
     * threshold of confidence
     * Type: Float
     * Required: No
     * Default: 0.50
     */
    void setConfidence(float value);
    float getConfidence() const;

    /**
     * path to labels file
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setLabels(const std::string& value);
    std::string getLabels() const;

    /**
     * which one to be classified
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setTarget(const std::string& value);
    std::string getTarget() const;

    Dnn_classify(int dnn_backend = 2, const std::string& model = "", const std::string& input = "", const std::string& output = "", const std::string& backend_configs = "", bool async = true, float confidence = 0.50, const std::string& labels = "", const std::string& target = "");
    virtual ~Dnn_classify();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int dnn_backend_;
    std::string model_;
    std::string input_;
    std::string output_;
    std::string backend_configs_;
    bool async_;
    float confidence_;
    std::string labels_;
    std::string target_;
};
