#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Areverse : public FilterBase {
public:
    /**
     * Reverse an audio clip.
     */
    Areverse();
    virtual ~Areverse();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
