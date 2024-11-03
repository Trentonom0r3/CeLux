#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Fade : public FilterBase {
public:
    /**
     * Fade in/out input video.
     */
    /**
     * set the fade direction
     * Aliases: t
     * Unit: type
     * Possible Values: in (0), out (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setType(int value);
    int getType() const;

    /**
     * Number of the first frame to which to apply the effect.
     * Aliases: s
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setStart_frame(int value);
    int getStart_frame() const;

    /**
     * Number of frames to which the effect should be applied.
     * Aliases: n
     * Type: Integer
     * Required: No
     * Default: 25
     */
    void setNb_frames(int value);
    int getNb_frames() const;

    /**
     * fade alpha if it is available on the input
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setAlpha(bool value);
    bool getAlpha() const;

    /**
     * Number of seconds of the beginning of the effect.
     * Aliases: st
     * Type: Duration
     * Required: No
     * Default: 0
     */
    void setStart_time(int64_t value);
    int64_t getStart_time() const;

    /**
     * Duration of the effect in seconds.
     * Aliases: d
     * Type: Duration
     * Required: No
     * Default: 0
     */
    void setDuration(int64_t value);
    int64_t getDuration() const;

    /**
     * set color
     * Aliases: c
     * Type: Color
     * Required: No
     * Default: black
     */
    void setColor(const std::string& value);
    std::string getColor() const;

    Fade(int type = 0, int start_frame = 0, int nb_frames = 25, bool alpha = false, int64_t start_time = 0ULL, int64_t duration = 0ULL, const std::string& color = "black");
    virtual ~Fade();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int type_;
    int start_frame_;
    int nb_frames_;
    bool alpha_;
    int64_t start_time_;
    int64_t duration_;
    std::string color_;
};
