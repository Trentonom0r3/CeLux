#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Extractplanes : public FilterBase {
public:
    /**
     * Extract planes as grayscale frames.
     */
    /**
     * set planes
     * Unit: flags
     * Possible Values: y (16), u (32), v (64), r (1), g (2), b (4), a (8)
     * Type: Flags
     * Required: No
     * Default: 1
     */
    void setPlanes(int value);
    int getPlanes() const;

    Extractplanes(int planes = 1);
    virtual ~Extractplanes();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int planes_;
};
