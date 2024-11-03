#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Asdr : public FilterBase {
public:
    /**
     * Measure Audio Signal-to-Distortion Ratio.
     */
    Asdr();
    virtual ~Asdr();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
