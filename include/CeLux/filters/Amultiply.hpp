#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Amultiply : public FilterBase {
public:
    /**
     * Multiply two audio streams.
     */
    Amultiply();
    virtual ~Amultiply();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
