#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Huesaturation : public FilterBase {
public:
    /**
     * Apply hue-saturation-intensity adjustments.
     */
    /**
     * set the hue shift
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setHue(float value);
    float getHue() const;

    /**
     * set the saturation shift
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setSaturation(float value);
    float getSaturation() const;

    /**
     * set the intensity shift
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setIntensity(float value);
    float getIntensity() const;

    /**
     * set colors range
     * Unit: colors
     * Possible Values: r (1), y (2), g (4), c (8), b (16), m (32), a (63)
     * Type: Flags
     * Required: No
     * Default: 63
     */
    void setColors(int value);
    int getColors() const;

    /**
     * set the filtering strength
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setStrength(float value);
    float getStrength() const;

    /**
     * set the red weight
     * Type: Float
     * Required: No
     * Default: 0.33
     */
    void setRw(float value);
    float getRw() const;

    /**
     * set the green weight
     * Type: Float
     * Required: No
     * Default: 0.33
     */
    void setGw(float value);
    float getGw() const;

    /**
     * set the blue weight
     * Type: Float
     * Required: No
     * Default: 0.33
     */
    void setBw(float value);
    float getBw() const;

    /**
     * set the preserve lightness
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setLightness(bool value);
    bool getLightness() const;

    Huesaturation(float hue = 0.00, float saturation = 0.00, float intensity = 0.00, int colors = 63, float strength = 1.00, float rw = 0.33, float gw = 0.33, float bw = 0.33, bool lightness = false);
    virtual ~Huesaturation();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float hue_;
    float saturation_;
    float intensity_;
    int colors_;
    float strength_;
    float rw_;
    float gw_;
    float bw_;
    bool lightness_;
};
