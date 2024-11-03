#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Gblur : public FilterBase {
public:
    /**
     * Apply Gaussian Blur filter.
     */
    /**
     * set sigma
     * Type: Float
     * Required: No
     * Default: 0.50
     */
    void setSigma(float value);
    float getSigma() const;

    /**
     * set number of steps
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setSteps(int value);
    int getSteps() const;

    /**
     * set planes to filter
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    /**
     * set vertical sigma
     * Type: Float
     * Required: No
     * Default: -1.00
     */
    void setSigmaV(float value);
    float getSigmaV() const;

    Gblur(float sigma = 0.50, int steps = 1, int planes = 15, float sigmaV = -1.00);
    virtual ~Gblur();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float sigma_;
    int steps_;
    int planes_;
    float sigmaV_;
};
