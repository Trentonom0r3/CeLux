#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Normalize : public FilterBase {
public:
    /**
     * Normalize RGB video.
     */
    /**
     * output color to which darkest input color is mapped
     * Type: Color
     * Required: No
     * Default: black
     */
    void setBlackpt(const std::string& value);
    std::string getBlackpt() const;

    /**
     * output color to which brightest input color is mapped
     * Type: Color
     * Required: No
     * Default: white
     */
    void setWhitept(const std::string& value);
    std::string getWhitept() const;

    /**
     * amount of temporal smoothing of the input range, to reduce flicker
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setSmoothing(int value);
    int getSmoothing() const;

    /**
     * proportion of independent to linked channel normalization
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setIndependence(float value);
    float getIndependence() const;

    /**
     * strength of filter, from no effect to full normalization
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setStrength(float value);
    float getStrength() const;

    Normalize(const std::string& blackpt = "black", const std::string& whitept = "white", int smoothing = 0, float independence = 1.00, float strength = 1.00);
    virtual ~Normalize();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string blackpt_;
    std::string whitept_;
    int smoothing_;
    float independence_;
    float strength_;
};
