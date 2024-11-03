#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Colorcorrect : public FilterBase {
public:
    /**
     * Adjust color white balance selectively for blacks and whites.
     */
    /**
     * set the red shadow spot
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setRl(float value);
    float getRl() const;

    /**
     * set the blue shadow spot
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setBl(float value);
    float getBl() const;

    /**
     * set the red highlight spot
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setRh(float value);
    float getRh() const;

    /**
     * set the blue highlight spot
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setBh(float value);
    float getBh() const;

    /**
     * set the amount of saturation
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setSaturation(float value);
    float getSaturation() const;

    /**
     * set the analyze mode
     * Unit: analyze
     * Possible Values: manual (0), average (1), minmax (2), median (3)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setAnalyze(int value);
    int getAnalyze() const;

    Colorcorrect(float rl = 0.00, float bl = 0.00, float rh = 0.00, float bh = 0.00, float saturation = 1.00, int analyze = 0);
    virtual ~Colorcorrect();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float rl_;
    float bl_;
    float rh_;
    float bh_;
    float saturation_;
    int analyze_;
};
