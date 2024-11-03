#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Swapuv : public FilterBase {
public:
    /**
     * Swap U and V components.
     */
    Swapuv();
    virtual ~Swapuv();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
