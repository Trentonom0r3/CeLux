#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Colorspace : public FilterBase {
public:
    /**
     * Convert between colorspaces.
     */
    /**
     * Set all color properties together
     * Unit: all
     * Possible Values: bt470m (1), bt470bg (2), bt601-6-525 (3), bt601-6-625 (4), bt709 (5), smpte170m (6), smpte240m (7), bt2020 (8)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setAll(int value);
    int getAll() const;

    /**
     * Output colorspace
     * Unit: csp
     * Possible Values: bt709 (1), fcc (4), bt470bg (5), smpte170m (6), smpte240m (7), ycgco (8), gbr (0), bt2020nc (9), bt2020ncl (9)
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setSpace(int value);
    int getSpace() const;

    /**
     * Output color range
     * Unit: rng
     * Possible Values: tv (1), mpeg (1), pc (2), jpeg (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setRange(int value);
    int getRange() const;

    /**
     * Output color primaries
     * Unit: prm
     * Possible Values: bt709 (1), bt470m (4), bt470bg (5), smpte170m (6), smpte240m (7), smpte428 (10), film (8), smpte431 (11), smpte432 (12), bt2020 (9), jedec-p22 (22), ebu3213 (22)
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setPrimaries(int value);
    int getPrimaries() const;

    /**
     * Output transfer characteristics
     * Unit: trc
     * Possible Values: bt709 (1), bt470m (4), gamma22 (4), bt470bg (5), gamma28 (5), smpte170m (6), smpte240m (7), linear (8), srgb (13), iec61966-2-1 (13), xvycc (11), iec61966-2-4 (11), bt2020-10 (14), bt2020-12 (15)
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setTrc(int value);
    int getTrc() const;

    /**
     * Output pixel format
     * Unit: fmt
     * Possible Values: yuv420p (0), yuv420p10 (62), yuv420p12 (123), yuv422p (4), yuv422p10 (64), yuv422p12 (127), yuv444p (5), yuv444p10 (68), yuv444p12 (131)
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setFormat(int value);
    int getFormat() const;

    /**
     * Ignore primary chromaticity and gamma correction
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setFast(bool value);
    bool getFast() const;

    /**
     * Dithering mode
     * Unit: dither
     * Possible Values: none (0), fsb (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setDither(int value);
    int getDither() const;

    /**
     * Whitepoint adaptation method
     * Unit: wpadapt
     * Possible Values: bradford (0), vonkries (1), identity (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setWpadapt(int value);
    int getWpadapt() const;

    /**
     * Set all input color properties together
     * Unit: all
     * Possible Values: bt470m (1), bt470bg (2), bt601-6-525 (3), bt601-6-625 (4), bt709 (5), smpte170m (6), smpte240m (7), bt2020 (8)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setIall(int value);
    int getIall() const;

    /**
     * Input colorspace
     * Unit: csp
     * Possible Values: bt709 (1), fcc (4), bt470bg (5), smpte170m (6), smpte240m (7), ycgco (8), gbr (0), bt2020nc (9), bt2020ncl (9)
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setIspace(int value);
    int getIspace() const;

    /**
     * Input color range
     * Unit: rng
     * Possible Values: tv (1), mpeg (1), pc (2), jpeg (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setIrange(int value);
    int getIrange() const;

    /**
     * Input color primaries
     * Unit: prm
     * Possible Values: bt709 (1), bt470m (4), bt470bg (5), smpte170m (6), smpte240m (7), smpte428 (10), film (8), smpte431 (11), smpte432 (12), bt2020 (9), jedec-p22 (22), ebu3213 (22)
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setIprimaries(int value);
    int getIprimaries() const;

    /**
     * Input transfer characteristics
     * Unit: trc
     * Possible Values: bt709 (1), bt470m (4), gamma22 (4), bt470bg (5), gamma28 (5), smpte170m (6), smpte240m (7), linear (8), srgb (13), iec61966-2-1 (13), xvycc (11), iec61966-2-4 (11), bt2020-10 (14), bt2020-12 (15)
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setItrc(int value);
    int getItrc() const;

    Colorspace(int all = 0, int space = 2, int range = 0, int primaries = 2, int trc = 2, int format = -1, bool fast = false, int dither = 0, int wpadapt = 0, int iall = 0, int ispace = 2, int irange = 0, int iprimaries = 2, int itrc = 2);
    virtual ~Colorspace();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int all_;
    int space_;
    int range_;
    int primaries_;
    int trc_;
    int format_;
    bool fast_;
    int dither_;
    int wpadapt_;
    int iall_;
    int ispace_;
    int irange_;
    int iprimaries_;
    int itrc_;
};
