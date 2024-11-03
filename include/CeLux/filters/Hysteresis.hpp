#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Hysteresis : public FilterBase {
public:
    /**
     * Grow first stream into second stream by connecting components.
     */
    /**
     * set planes
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    /**
     * set threshold
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setThreshold(int value);
    int getThreshold() const;

    Hysteresis(int planes = 15, int threshold = 0);
    virtual ~Hysteresis();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int planes_;
    int threshold_;
};
