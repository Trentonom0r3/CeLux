#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Dnn_detect : public FilterBase {
public:
    /**
     * Apply DNN detect filter to the input.
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
     * DNN detection model type
     * Unit: model_type
     * Possible Values: ssd (0), yolo (1), yolov3 (2), yolov4 (3)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setModel_type(int value);
    int getModel_type() const;

    /**
     * cell width
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setCell_w(int value);
    int getCell_w() const;

    /**
     * cell height
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setCell_h(int value);
    int getCell_h() const;

    /**
     * The number of class
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setNb_classes(int value);
    int getNb_classes() const;

    /**
     * anchors, splited by '&'
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setAnchors(const std::string& value);
    std::string getAnchors() const;

    Dnn_detect(int dnn_backend = 2, const std::string& model = "", const std::string& input = "", const std::string& output = "", const std::string& backend_configs = "", bool async = true, float confidence = 0.50, const std::string& labels = "", int model_type = 0, int cell_w = 0, int cell_h = 0, int nb_classes = 0, const std::string& anchors = "");
    virtual ~Dnn_detect();

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
    int model_type_;
    int cell_w_;
    int cell_h_;
    int nb_classes_;
    std::string anchors_;
};
