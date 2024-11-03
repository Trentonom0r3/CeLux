#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Sidedata : public FilterBase {
public:
    /**
     * Manipulate video frame side data.
     */
    /**
     * set a mode of operation
     * Unit: mode
     * Possible Values: select (0), delete (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    /**
     * set side data type
     * Unit: type
     * Possible Values: PANSCAN (0), A53_CC (1), STEREO3D (2), MATRIXENCODING (3), DOWNMIX_INFO (4), REPLAYGAIN (5), DISPLAYMATRIX (6), AFD (7), MOTION_VECTORS (8), SKIP_SAMPLES (9), AUDIO_SERVICE_TYPE (10), MASTERING_DISPLAY_METADATA (11), GOP_TIMECODE (12), SPHERICAL (13), CONTENT_LIGHT_LEVEL (14), ICC_PROFILE (15), S12M_TIMECOD (16), DYNAMIC_HDR_PLUS (17), REGIONS_OF_INTEREST (18), DETECTION_BOUNDING_BOXES (22), SEI_UNREGISTERED (20)
     * Type: Integer
     * Required: No
     * Default: -1
     */
    void setType(int value);
    int getType() const;

    Sidedata(int mode = 0, int type = -1);
    virtual ~Sidedata();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int mode_;
    int type_;
};
