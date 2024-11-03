#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Copy : public FilterBase {
public:
    /**
     * Copy the input video unchanged to the output.
     */
    Copy();
    virtual ~Copy();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
