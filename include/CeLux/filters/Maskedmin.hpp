#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Maskedmin : public FilterBase {
public:
    /**
     * Apply filtering with minimum difference of two streams.
     */
    /**
     * set planes
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    Maskedmin(int planes = 15);
    virtual ~Maskedmin();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int planes_;
};
