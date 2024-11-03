#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Apsnr : public FilterBase {
public:
    /**
     * Measure Audio Peak Signal-to-Noise Ratio.
     */
    Apsnr();
    virtual ~Apsnr();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
