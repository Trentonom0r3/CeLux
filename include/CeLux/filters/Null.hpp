#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Null : public FilterBase {
public:
    /**
     * Pass the source unchanged to the output.
     */
    Null();
    virtual ~Null();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
