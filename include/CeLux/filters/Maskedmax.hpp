#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Maskedmax : public FilterBase {
public:
    /**
     * Apply filtering with maximum difference of two streams.
     */
    /**
     * set planes
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    Maskedmax(int planes = 15);
    virtual ~Maskedmax();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int planes_;
};
