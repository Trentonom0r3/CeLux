#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Msad : public FilterBase {
public:
    /**
     * Calculate the MSAD between two video streams.
     */
    Msad();
    virtual ~Msad();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
