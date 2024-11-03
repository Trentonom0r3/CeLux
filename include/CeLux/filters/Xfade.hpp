#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Xfade : public FilterBase {
public:
    /**
     * Cross fade one video with another video.
     */
    /**
     * set cross fade transition
     * Unit: transition
     * Possible Values: custom (-1), fade (0), wipeleft (1), wiperight (2), wipeup (3), wipedown (4), slideleft (5), slideright (6), slideup (7), slidedown (8), circlecrop (9), rectcrop (10), distance (11), fadeblack (12), fadewhite (13), radial (14), smoothleft (15), smoothright (16), smoothup (17), smoothdown (18), circleopen (19), circleclose (20), vertopen (21), vertclose (22), horzopen (23), horzclose (24), dissolve (25), pixelize (26), diagtl (27), diagtr (28), diagbl (29), diagbr (30), hlslice (31), hrslice (32), vuslice (33), vdslice (34), hblur (35), fadegrays (36), wipetl (37), wipetr (38), wipebl (39), wipebr (40), squeezeh (41), squeezev (42), zoomin (43), fadefast (44), fadeslow (45), hlwind (46), hrwind (47), vuwind (48), vdwind (49), coverleft (50), coverright (51), coverup (52), coverdown (53), revealleft (54), revealright (55), revealup (56), revealdown (57)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setTransition(int value);
    int getTransition() const;

    /**
     * set cross fade duration
     * Type: Duration
     * Required: No
     * Default: 1000000
     */
    void setDuration(int64_t value);
    int64_t getDuration() const;

    /**
     * set cross fade start relative to first input stream
     * Type: Duration
     * Required: No
     * Default: 0
     */
    void setOffset(int64_t value);
    int64_t getOffset() const;

    /**
     * set expression for custom transition
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setExpr(const std::string& value);
    std::string getExpr() const;

    Xfade(int transition = 0, int64_t duration = 1000000ULL, int64_t offset = 0ULL, const std::string& expr = "");
    virtual ~Xfade();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int transition_;
    int64_t duration_;
    int64_t offset_;
    std::string expr_;
};
