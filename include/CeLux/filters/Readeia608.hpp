#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Readeia608 : public FilterBase {
public:
    /**
     * Read EIA-608 Closed Caption codes from input video and write them to frame metadata.
     */
    /**
     * set from which line to scan for codes
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setScan_min(int value);
    int getScan_min() const;

    /**
     * set to which line to scan for codes
     * Type: Integer
     * Required: No
     * Default: 29
     */
    void setScan_max(int value);
    int getScan_max() const;

    /**
     * set ratio of width reserved for sync code detection
     * Type: Float
     * Required: No
     * Default: 0.27
     */
    void setSpw(float value);
    float getSpw() const;

    /**
     * check and apply parity bit
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setChp(bool value);
    bool getChp() const;

    /**
     * lowpass line prior to processing
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setLp(bool value);
    bool getLp() const;

    Readeia608(int scan_min = 0, int scan_max = 29, float spw = 0.27, bool chp = false, bool lp = true);
    virtual ~Readeia608();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int scan_min_;
    int scan_max_;
    float spw_;
    bool chp_;
    bool lp_;
};
