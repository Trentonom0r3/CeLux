#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Vif : public FilterBase {
public:
    /**
     * Calculate the VIF between two video streams.
     */
    Vif();
    virtual ~Vif();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
