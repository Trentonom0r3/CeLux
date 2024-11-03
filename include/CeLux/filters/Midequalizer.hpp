#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Midequalizer : public FilterBase {
public:
    /**
     * Apply Midway Equalization.
     */
    /**
     * set planes
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    Midequalizer(int planes = 15);
    virtual ~Midequalizer();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int planes_;
};
