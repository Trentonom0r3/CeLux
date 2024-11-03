#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Ashowinfo : public FilterBase {
public:
    /**
     * Show textual information for each audio frame.
     */
    Ashowinfo();
    virtual ~Ashowinfo();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
