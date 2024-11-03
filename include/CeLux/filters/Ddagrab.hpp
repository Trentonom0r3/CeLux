#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Ddagrab : public FilterBase {
public:
    /**
     * Grab Windows Desktop images using Desktop Duplication API
     */
    /**
     * dda output index to capture
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setOutput_idx(int value);
    int getOutput_idx() const;

    /**
     * draw the mouse pointer
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setDraw_mouse(bool value);
    bool getDraw_mouse() const;

    /**
     * set video frame rate
     * Type: Video Rate
     * Required: No
     * Default: 40570.7
     */
    void setFramerate(const std::pair<int, int>& value);
    std::pair<int, int> getFramerate() const;

    /**
     * set video frame size
     * Type: Image Size
     * Required: Yes
     * Default: No Default
     */
    void setVideo_size(const std::pair<int, int>& value);
    std::pair<int, int> getVideo_size() const;

    /**
     * capture area x offset
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setOffset_x(int value);
    int getOffset_x() const;

    /**
     * capture area y offset
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setOffset_y(int value);
    int getOffset_y() const;

    /**
     * desired output format
     * Unit: output_fmt
     * Possible Values: auto (0), 8bit (87), bgra (87), 10bit (24), x2bgr10 (24), 16bit (10), rgbaf16 (10)
     * Type: Integer
     * Required: No
     * Default: 87
     */
    void setOutput_fmt(int value);
    int getOutput_fmt() const;

    /**
     * don't error on fallback to default 8 Bit format
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setAllow_fallback(bool value);
    bool getAllow_fallback() const;

    /**
     * exclude BGRA from format list (experimental, discouraged by Microsoft)
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setForce_fmt(bool value);
    bool getForce_fmt() const;

    /**
     * duplicate frames to maintain framerate
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setDup_frames(bool value);
    bool getDup_frames() const;

    Ddagrab(int output_idx = 0, bool draw_mouse = true, std::pair<int, int> framerate = std::make_pair<int, int>(0, 1), std::pair<int, int> video_size = std::make_pair<int, int>(0, 1), int offset_x = 0, int offset_y = 0, int output_fmt = 87, bool allow_fallback = false, bool force_fmt = false, bool dup_frames = true);
    virtual ~Ddagrab();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int output_idx_;
    bool draw_mouse_;
    std::pair<int, int> framerate_;
    std::pair<int, int> video_size_;
    int offset_x_;
    int offset_y_;
    int output_fmt_;
    bool allow_fallback_;
    bool force_fmt_;
    bool dup_frames_;
};
