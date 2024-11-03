#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Aderivative : public FilterBase {
public:
    /**
     * Compute derivative of input audio.
     */
    Aderivative();
    virtual ~Aderivative();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
