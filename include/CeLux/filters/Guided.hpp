#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Guided : public FilterBase {
public:
    /**
     * Apply Guided filter.
     */
    /**
     * set the box radius
     * Type: Integer
     * Required: No
     * Default: 3
     */
    void setRadius(int value);
    int getRadius() const;

    /**
     * set the regularization parameter (with square)
     * Type: Float
     * Required: No
     * Default: 0.01
     */
    void setEps(float value);
    float getEps() const;

    /**
     * set filtering mode (0: basic mode; 1: fast mode)
     * Unit: mode
     * Possible Values: basic (0), fast (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    /**
     * subsampling ratio for fast mode
     * Type: Integer
     * Required: No
     * Default: 4
     */
    void setSub(int value);
    int getSub() const;

    /**
     * set guidance mode (0: off mode; 1: on mode)
     * Unit: guidance
     * Possible Values: off (0), on (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setGuidance(int value);
    int getGuidance() const;

    /**
     * set planes to filter
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setPlanes(int value);
    int getPlanes() const;

    Guided(int radius = 3, float eps = 0.01, int mode = 0, int sub = 4, int guidance = 0, int planes = 1);
    virtual ~Guided();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int radius_;
    float eps_;
    int mode_;
    int sub_;
    int guidance_;
    int planes_;
};
