#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Xcorrelate : public FilterBase {
public:
    /**
     * Cross-correlate first video stream with second video stream.
     */
    /**
     * set planes to cross-correlate
     * Type: Integer
     * Required: No
     * Default: 7
     */
    void setPlanes(int value);
    int getPlanes() const;

    /**
     * when to process secondary frame
     * Unit: impulse
     * Possible Values: first (0), all (1)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setSecondary(int value);
    int getSecondary() const;

    Xcorrelate(int planes = 7, int secondary = 1);
    virtual ~Xcorrelate();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int planes_;
    int secondary_;
};
