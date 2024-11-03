#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Reverse : public FilterBase {
public:
    /**
     * Reverse a clip.
     */
    Reverse();
    virtual ~Reverse();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
