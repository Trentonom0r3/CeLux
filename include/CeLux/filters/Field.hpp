#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Field : public FilterBase {
public:
    /**
     * Extract a field from the input video.
     */
    /**
     * set field type (top or bottom)
     * Unit: field_type
     * Possible Values: top (0), bottom (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setType(int value);
    int getType() const;

    Field(int type = 0);
    virtual ~Field();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int type_;
};
