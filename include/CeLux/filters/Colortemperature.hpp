#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Colortemperature : public FilterBase {
public:
    /**
     * Adjust color temperature of video.
     */
    /**
     * set the temperature in Kelvin
     * Type: Float
     * Required: No
     * Default: 6500.00
     */
    void setTemperature(float value);
    float getTemperature() const;

    /**
     * set the mix with filtered output
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setMix(float value);
    float getMix() const;

    /**
     * set the amount of preserving lightness
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setPl(float value);
    float getPl() const;

    Colortemperature(float temperature = 6500.00, float mix = 1.00, float pl = 0.00);
    virtual ~Colortemperature();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float temperature_;
    float mix_;
    float pl_;
};
