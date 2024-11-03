#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Colorkey : public FilterBase {
public:
    /**
     * Turns a certain color into transparency. Operates on RGB colors.
     */
    /**
     * set the colorkey key color
     * Type: Color
     * Required: No
     * Default: black
     */
    void setColor(const std::string& value);
    std::string getColor() const;

    /**
     * set the colorkey similarity value
     * Type: Float
     * Required: No
     * Default: 0.01
     */
    void setSimilarity(float value);
    float getSimilarity() const;

    /**
     * set the colorkey key blend value
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setBlend(float value);
    float getBlend() const;

    Colorkey(const std::string& color = "black", float similarity = 0.01, float blend = 0.00);
    virtual ~Colorkey();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string color_;
    float similarity_;
    float blend_;
};
