#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Framepack : public FilterBase {
public:
    /**
     * Generate a frame packed stereoscopic video.
     */
    /**
     * Frame pack output format
     * Unit: format
     * Possible Values: sbs (1), tab (2), frameseq (3), lines (6), columns (7)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setFormat(int value);
    int getFormat() const;

    Framepack(int format = 1);
    virtual ~Framepack();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int format_;
};
