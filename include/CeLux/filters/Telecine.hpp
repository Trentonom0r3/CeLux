#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Telecine : public FilterBase {
public:
    /**
     * Apply a telecine pattern.
     */
    /**
     * select first field
     * Unit: field
     * Possible Values: top (0), t (0), bottom (1), b (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setFirst_field(int value);
    int getFirst_field() const;

    /**
     * pattern that describe for how many fields a frame is to be displayed
     * Type: String
     * Required: No
     * Default: 23
     */
    void setPattern(const std::string& value);
    std::string getPattern() const;

    Telecine(int first_field = 0, const std::string& pattern = "23");
    virtual ~Telecine();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int first_field_;
    std::string pattern_;
};
