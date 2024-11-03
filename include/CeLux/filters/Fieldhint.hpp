#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Fieldhint : public FilterBase {
public:
    /**
     * Field matching using hints.
     */
    /**
     * set hint file
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setHint(const std::string& value);
    std::string getHint() const;

    /**
     * set hint mode
     * Unit: mode
     * Possible Values: absolute (0), relative (1), pattern (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    Fieldhint(const std::string& hint = "", int mode = 0);
    virtual ~Fieldhint();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string hint_;
    int mode_;
};
