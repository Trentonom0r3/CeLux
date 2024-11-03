#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Alphamerge : public FilterBase {
public:
    /**
     * Copy the luma value of the second input into the alpha channel of the first input.
     */
    Alphamerge();
    virtual ~Alphamerge();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
