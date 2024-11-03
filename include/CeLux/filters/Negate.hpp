#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Negate : public FilterBase {
public:
    /**
     * Negate input video.
     */
    /**
     * set components to negate
     * Unit: flags
     * Possible Values: y (16), u (32), v (64), r (1), g (2), b (4), a (8)
     * Type: Flags
     * Required: No
     * Default: 119
     */
    void setComponents(int value);
    int getComponents() const;

    /**
     * 
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setNegate_alpha(bool value);
    bool getNegate_alpha() const;

    Negate(int components = 119, bool negate_alpha = false);
    virtual ~Negate();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int components_;
    bool negate_alpha_;
};
