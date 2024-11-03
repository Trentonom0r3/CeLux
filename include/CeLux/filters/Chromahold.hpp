#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Chromahold : public FilterBase {
public:
    /**
     * Turns a certain color range into gray.
     */
    /**
     * set the chromahold key color
     * Type: Color
     * Required: No
     * Default: black
     */
    void setColor(const std::string& value);
    std::string getColor() const;

    /**
     * set the chromahold similarity value
     * Type: Float
     * Required: No
     * Default: 0.01
     */
    void setSimilarity(float value);
    float getSimilarity() const;

    /**
     * set the chromahold blend value
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setBlend(float value);
    float getBlend() const;

    /**
     * color parameter is in yuv instead of rgb
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setYuv(bool value);
    bool getYuv() const;

    Chromahold(const std::string& color = "black", float similarity = 0.01, float blend = 0.00, bool yuv = false);
    virtual ~Chromahold();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string color_;
    float similarity_;
    float blend_;
    bool yuv_;
};
