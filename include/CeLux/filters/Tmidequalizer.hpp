#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Tmidequalizer : public FilterBase {
public:
    /**
     * Apply Temporal Midway Equalization.
     */
    /**
     * set radius
     * Type: Integer
     * Required: No
     * Default: 5
     */
    void setRadius(int value);
    int getRadius() const;

    /**
     * set sigma
     * Type: Float
     * Required: No
     * Default: 0.50
     */
    void setSigma(float value);
    float getSigma() const;

    /**
     * set planes
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    Tmidequalizer(int radius = 5, float sigma = 0.50, int planes = 15);
    virtual ~Tmidequalizer();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int radius_;
    float sigma_;
    int planes_;
};
