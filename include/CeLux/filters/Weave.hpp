#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Weave : public FilterBase {
public:
    /**
     * Weave input video fields into frames.
     */
    /**
     * set first field
     * Unit: field
     * Possible Values: top (0), t (0), bottom (1), b (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setFirst_field(int value);
    int getFirst_field() const;

    Weave(int first_field = 0);
    virtual ~Weave();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int first_field_;
};
