#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Corr : public FilterBase {
public:
    /**
     * Calculate the correlation between two video streams.
     */
    Corr();
    virtual ~Corr();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
