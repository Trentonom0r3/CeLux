#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Despill : public FilterBase {
public:
    /**
     * Despill video.
     */
    /**
     * set the screen type
     * Unit: type
     * Possible Values: green (0), blue (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setType(int value);
    int getType() const;

    /**
     * set the spillmap mix
     * Type: Float
     * Required: No
     * Default: 0.50
     */
    void setMix(float value);
    float getMix() const;

    /**
     * set the spillmap expand
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setExpand(float value);
    float getExpand() const;

    /**
     * set red scale
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setRed(float value);
    float getRed() const;

    /**
     * set green scale
     * Type: Float
     * Required: No
     * Default: -1.00
     */
    void setGreen(float value);
    float getGreen() const;

    /**
     * set blue scale
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setBlue(float value);
    float getBlue() const;

    /**
     * set brightness
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setBrightness(float value);
    float getBrightness() const;

    /**
     * change alpha component
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setAlpha(bool value);
    bool getAlpha() const;

    Despill(int type = 0, float mix = 0.50, float expand = 0.00, float red = 0.00, float green = -1.00, float blue = 0.00, float brightness = 0.00, bool alpha = false);
    virtual ~Despill();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int type_;
    float mix_;
    float expand_;
    float red_;
    float green_;
    float blue_;
    float brightness_;
    bool alpha_;
};
