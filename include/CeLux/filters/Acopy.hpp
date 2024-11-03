#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Acopy : public FilterBase {
public:
    /**
     * Copy the input audio unchanged to the output.
     */
    Acopy();
    virtual ~Acopy();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
