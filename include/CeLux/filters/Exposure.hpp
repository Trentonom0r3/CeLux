#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Exposure : public FilterBase {
public:
    /**
     * Adjust exposure of the video stream.
     */
    /**
     * set the exposure correction
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setExposure(float value);
    float getExposure() const;

    /**
     * set the black level correction
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setBlack(float value);
    float getBlack() const;

    Exposure(float exposure = 0.00, float black = 0.00);
    virtual ~Exposure();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float exposure_;
    float black_;
};
