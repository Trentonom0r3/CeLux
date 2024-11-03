#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Fieldorder : public FilterBase {
public:
    /**
     * Set the field order.
     */
    /**
     * output field order
     * Unit: order
     * Possible Values: bff (0), tff (1)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setOrder(int value);
    int getOrder() const;

    Fieldorder(int order = 1);
    virtual ~Fieldorder();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int order_;
};
