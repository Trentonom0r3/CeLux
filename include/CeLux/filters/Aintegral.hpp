#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Aintegral : public FilterBase {
public:
    /**
     * Compute integral of input audio.
     */
    Aintegral();
    virtual ~Aintegral();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
