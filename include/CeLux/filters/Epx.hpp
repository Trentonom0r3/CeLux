#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Epx : public FilterBase {
public:
    /**
     * Scale the input using EPX algorithm.
     */
    /**
     * set scale factor
     * Type: Integer
     * Required: No
     * Default: 3
     */
    void setScaleFactor(int value);
    int getScaleFactor() const;

    Epx(int scaleFactor = 3);
    virtual ~Epx();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int scaleFactor_;
};
