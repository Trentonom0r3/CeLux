#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Limiter : public FilterBase {
public:
    /**
     * Limit pixels components to the specified range.
     */
    /**
     * set min value
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMin(int value);
    int getMin() const;

    /**
     * set max value
     * Type: Integer
     * Required: No
     * Default: 65535
     */
    void setMax(int value);
    int getMax() const;

    /**
     * set planes
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    Limiter(int min = 0, int max = 65535, int planes = 15);
    virtual ~Limiter();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int min_;
    int max_;
    int planes_;
};
