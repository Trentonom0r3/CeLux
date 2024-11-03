#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Colorize : public FilterBase {
public:
    /**
     * Overlay a solid color on the video stream.
     */
    /**
     * set the hue
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setHue(float value);
    float getHue() const;

    /**
     * set the saturation
     * Type: Float
     * Required: No
     * Default: 0.50
     */
    void setSaturation(float value);
    float getSaturation() const;

    /**
     * set the lightness
     * Type: Float
     * Required: No
     * Default: 0.50
     */
    void setLightness(float value);
    float getLightness() const;

    /**
     * set the mix of source lightness
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setMix(float value);
    float getMix() const;

    Colorize(float hue = 0.00, float saturation = 0.50, float lightness = 0.50, float mix = 1.00);
    virtual ~Colorize();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float hue_;
    float saturation_;
    float lightness_;
    float mix_;
};
