#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Convolve : public FilterBase {
public:
    /**
     * Convolve first video stream with second video stream.
     */
    /**
     * set planes to convolve
     * Type: Integer
     * Required: No
     * Default: 7
     */
    void setPlanes(int value);
    int getPlanes() const;

    /**
     * when to process impulses
     * Unit: impulse
     * Possible Values: first (0), all (1)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setImpulse(int value);
    int getImpulse() const;

    /**
     * set noise
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setNoise(float value);
    float getNoise() const;

    Convolve(int planes = 7, int impulse = 1, float noise = 0.00);
    virtual ~Convolve();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int planes_;
    int impulse_;
    float noise_;
};
