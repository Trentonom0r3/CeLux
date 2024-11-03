#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Asisdr : public FilterBase {
public:
    /**
     * Measure Audio Scale-Invariant Signal-to-Distortion Ratio.
     */
    Asisdr();
    virtual ~Asisdr();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
