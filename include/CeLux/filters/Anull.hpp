#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Anull : public FilterBase {
public:
    /**
     * Pass the source unchanged to the output.
     */
    Anull();
    virtual ~Anull();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
