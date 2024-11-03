#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Hqx : public FilterBase {
public:
    /**
     * Scale the input by 2, 3 or 4 using the hq*x magnification algorithm.
     */
    /**
     * set scale factor
     * Type: Integer
     * Required: No
     * Default: 3
     */
    void setScaleFactor(int value);
    int getScaleFactor() const;

    Hqx(int scaleFactor = 3);
    virtual ~Hqx();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int scaleFactor_;
};
