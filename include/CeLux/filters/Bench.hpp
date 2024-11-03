#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Bench : public FilterBase {
public:
    /**
     * Benchmark part of a filtergraph.
     */
    /**
     * set action
     * Unit: action
     * Possible Values: start (0), stop (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setAction(int value);
    int getAction() const;

    Bench(int action = 0);
    virtual ~Bench();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int action_;
};
