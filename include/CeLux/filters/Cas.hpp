#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Cas : public FilterBase {
public:
    /**
     * Contrast Adaptive Sharpen.
     */
    /**
     * set the sharpening strength
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setStrength(float value);
    float getStrength() const;

    /**
     * set what planes to filter
     * Type: Flags
     * Required: No
     * Default: 7
     */
    void setPlanes(int value);
    int getPlanes() const;

    Cas(float strength = 0.00, int planes = 7);
    virtual ~Cas();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float strength_;
    int planes_;
};
