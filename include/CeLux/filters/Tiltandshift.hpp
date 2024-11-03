#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Tiltandshift : public FilterBase {
public:
    /**
     * Generate a tilt-and-shift'd video.
     */
    /**
     * Tilt the video horizontally while shifting
     * Unit: tilt
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setTilt(int value);
    int getTilt() const;

    /**
     * Action at the start of input
     * Unit: start
     * Possible Values: none (0), frame (1), black (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setStart(int value);
    int getStart() const;

    /**
     * Action at the end of input
     * Unit: end
     * Possible Values: none (0), frame (1), black (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setEnd(int value);
    int getEnd() const;

    /**
     * Number of columns to hold at the start of the video
     * Unit: hold
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setHold(int value);
    int getHold() const;

    /**
     * Number of columns to pad at the end of the video
     * Unit: pad
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setPad(int value);
    int getPad() const;

    Tiltandshift(int tilt = 1, int start = 0, int end = 0, int hold = 0, int pad = 0);
    virtual ~Tiltandshift();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int tilt_;
    int start_;
    int end_;
    int hold_;
    int pad_;
};
