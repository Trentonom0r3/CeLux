#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Colorhold : public FilterBase {
public:
    /**
     * Turns a certain color range into gray. Operates on RGB colors.
     */
    /**
     * set the colorhold key color
     * Type: Color
     * Required: No
     * Default: black
     */
    void setColor(const std::string& value);
    std::string getColor() const;

    /**
     * set the colorhold similarity value
     * Type: Float
     * Required: No
     * Default: 0.01
     */
    void setSimilarity(float value);
    float getSimilarity() const;

    /**
     * set the colorhold blend value
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setBlend(float value);
    float getBlend() const;

    Colorhold(const std::string& color = "black", float similarity = 0.01, float blend = 0.00);
    virtual ~Colorhold();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string color_;
    float similarity_;
    float blend_;
};
