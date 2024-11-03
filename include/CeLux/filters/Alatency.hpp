#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Alatency : public FilterBase {
public:
    /**
     * Report audio filtering latency.
     */
    Alatency();
    virtual ~Alatency();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
