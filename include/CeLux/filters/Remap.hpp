#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Remap : public FilterBase {
public:
    /**
     * Remap pixels.
     */
    /**
     * set output format
     * Unit: format
     * Possible Values: color (0), gray (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setFormat(int value);
    int getFormat() const;

    /**
     * set the color of the unmapped pixels
     * Type: Color
     * Required: No
     * Default: black
     */
    void setFill(const std::string& value);
    std::string getFill() const;

    Remap(int format = 0, const std::string& fill = "black");
    virtual ~Remap();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int format_;
    std::string fill_;
};
