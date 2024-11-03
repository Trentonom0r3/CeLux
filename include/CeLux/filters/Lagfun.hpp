#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Lagfun : public FilterBase {
public:
    /**
     * Slowly update darker pixels.
     */
    /**
     * set decay
     * Type: Float
     * Required: No
     * Default: 0.95
     */
    void setDecay(float value);
    float getDecay() const;

    /**
     * set what planes to filter
     * Type: Flags
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    Lagfun(float decay = 0.95, int planes = 15);
    virtual ~Lagfun();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float decay_;
    int planes_;
};
