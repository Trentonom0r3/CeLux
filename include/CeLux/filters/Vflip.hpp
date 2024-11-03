#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Vflip : public FilterBase {
public:
    /**
     * Flip the input video vertically.
     */
    Vflip();
    virtual ~Vflip();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
