#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Latency : public FilterBase {
public:
    /**
     * Report video filtering latency.
     */
    Latency();
    virtual ~Latency();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
