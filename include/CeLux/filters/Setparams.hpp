#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Setparams : public FilterBase {
public:
    /**
     * Force field, or color property for the output video frame.
     */
    /**
     * select interlace mode
     * Unit: mode
     * Possible Values: auto (-1), bff (0), tff (1), prog (2)
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setField_mode(int value);
    int getField_mode() const;

    /**
     * select color range
     * Unit: range
     * Possible Values: auto (-1), unspecified (0), unknown (0), limited (1), tv (1), mpeg (1), full (2), pc (2), jpeg (2)
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setRange(int value);
    int getRange() const;

    /**
     * select color primaries
     * Unit: color_primaries
     * Possible Values: auto (-1), bt709 (1), unknown (2), bt470m (4), bt470bg (5), smpte170m (6), smpte240m (7), film (8), bt2020 (9), smpte428 (10), smpte431 (11), smpte432 (12), jedec-p22 (22), ebu3213 (22)
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setColor_primaries(int value);
    int getColor_primaries() const;

    /**
     * select color transfer
     * Unit: color_trc
     * Possible Values: auto (-1), bt709 (1), unknown (2), bt470m (4), bt470bg (5), smpte170m (6), smpte240m (7), linear (8), log100 (9), log316 (10), iec61966-2-4 (11), bt1361e (12), iec61966-2-1 (13), bt2020-10 (14), bt2020-12 (15), smpte2084 (16), smpte428 (17), arib-std-b67 (18)
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setColor_trc(int value);
    int getColor_trc() const;

    /**
     * select colorspace
     * Unit: colorspace
     * Possible Values: auto (-1), gbr (0), bt709 (1), unknown (2), fcc (4), bt470bg (5), smpte170m (6), smpte240m (7), ycgco (8), bt2020nc (9), bt2020c (10), smpte2085 (11), chroma-derived-nc (12), chroma-derived-c (13), ictcp (14)
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setColorspace(int value);
    int getColorspace() const;

    Setparams(int field_mode = -1, int range = -1, int color_primaries = -1, int color_trc = -1, int colorspace = -1);
    virtual ~Setparams();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int field_mode_;
    int range_;
    int color_primaries_;
    int color_trc_;
    int colorspace_;
};
