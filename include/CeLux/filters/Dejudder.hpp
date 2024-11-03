#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Dejudder : public FilterBase {
public:
    /**
     * Remove judder produced by pullup.
     */
    /**
     * set the length of the cycle to use for dejuddering
     * Type: Integer
     * Required: No
     * Default: 4
     */
    void setCycle(int value);
    int getCycle() const;

    Dejudder(int cycle = 4);
    virtual ~Dejudder();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int cycle_;
};
