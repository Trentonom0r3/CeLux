#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Hue : public FilterBase {
public:
    /**
     * Adjust the hue and saturation of the input video.
     */
    /**
     * set the hue angle degrees expression
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setHueAngleDegrees(const std::string& value);
    std::string getHueAngleDegrees() const;

    /**
     * set the saturation expression
     * Type: String
     * Required: No
     * Default: 1
     */
    void setSaturation(const std::string& value);
    std::string getSaturation() const;

    /**
     * set the hue angle radians expression
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setHueAngleRadians(const std::string& value);
    std::string getHueAngleRadians() const;

    /**
     * set the brightness expression
     * Type: String
     * Required: No
     * Default: 0
     */
    void setBrightness(const std::string& value);
    std::string getBrightness() const;

    Hue(const std::string& hueAngleDegrees = "", const std::string& saturation = "1", const std::string& hueAngleRadians = "", const std::string& brightness = "0");
    virtual ~Hue();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string hueAngleDegrees_;
    std::string saturation_;
    std::string hueAngleRadians_;
    std::string brightness_;
};
