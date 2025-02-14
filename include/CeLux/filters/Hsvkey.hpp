#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Hsvkey : public FilterBase {
public:
    /**
     * Turns a certain HSV range into transparency. Operates on YUV colors.
     */
    /**
     * set the hue value
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setHue(float value);
    float getHue() const;

    /**
     * set the saturation value
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setSat(float value);
    float getSat() const;

    /**
     * set the value value
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setVal(float value);
    float getVal() const;

    /**
     * set the hsvkey similarity value
     * Type: Float
     * Required: No
     * Default: 0.01
     */
    void setSimilarity(float value);
    float getSimilarity() const;

    /**
     * set the hsvkey blend value
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setBlend(float value);
    float getBlend() const;

    Hsvkey(float hue = 0.00, float sat = 0.00, float val = 0.00, float similarity = 0.01, float blend = 0.00);
    virtual ~Hsvkey();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float hue_;
    float sat_;
    float val_;
    float similarity_;
    float blend_;
};
