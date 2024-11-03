#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Setfield : public FilterBase {
public:
    /**
     * Force field for the output video frame.
     */
    /**
     * select interlace mode
     * Unit: mode
     * Possible Values: auto (-1), bff (0), tff (1), prog (2)
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setMode(int value);
    int getMode() const;

    Setfield(int mode = -1);
    virtual ~Setfield();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int mode_;
};
