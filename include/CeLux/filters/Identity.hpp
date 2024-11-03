#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Identity : public FilterBase {
public:
    /**
     * Calculate the Identity between two video streams.
     */
    Identity();
    virtual ~Identity();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
