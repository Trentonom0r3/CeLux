#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Blockdetect : public FilterBase {
public:
    /**
     * Blockdetect filter.
     */
    /**
     * Minimum period to search for
     * Type: Integer
     * Required: No
     * Default: 3
     */
    void setPeriod_min(int value);
    int getPeriod_min() const;

    /**
     * Maximum period to search for
     * Type: Integer
     * Required: No
     * Default: 24
     */
    void setPeriod_max(int value);
    int getPeriod_max() const;

    /**
     * set planes to filter
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setPlanes(int value);
    int getPlanes() const;

    Blockdetect(int period_min = 3, int period_max = 24, int planes = 1);
    virtual ~Blockdetect();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int period_min_;
    int period_max_;
    int planes_;
};
