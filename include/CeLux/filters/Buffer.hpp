#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Buffer : public FilterBase {
public:
    /**
     * Buffer video frames, and make them accessible to the filterchain.
     */
    /**
     * 
     * Aliases: width
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setHeight(int value);
    int getHeight() const;

    /**
     * 
     * Type: Image Size
     * Required: Yes
     * Default: No Default
     */
    void setVideo_size(const std::pair<int, int>& value);
    std::pair<int, int> getVideo_size() const;

    /**
     * 
     * Type: Pixel Format
     * Required: Yes
     * Default: No Default
     */
    void setPix_fmt(const std::string& value);
    std::string getPix_fmt() const;

    /**
     * sample aspect ratio
     * Aliases: sar
     * Type: Rational
     * Required: Yes
     * Default: No Default
     */
    void setPixel_aspect(const std::pair<int, int>& value);
    std::pair<int, int> getPixel_aspect() const;

    /**
     * 
     * Aliases: time_base
     * Type: Rational
     * Required: Yes
     * Default: No Default
     */
    void setFrame_rate(const std::pair<int, int>& value);
    std::pair<int, int> getFrame_rate() const;

    /**
     * select colorspace
     * Unit: colorspace
     * Possible Values: gbr (0), bt709 (1), unknown (2), fcc (4), bt470bg (5), smpte170m (6), smpte240m (7), ycgco (8), bt2020nc (9), bt2020c (10), smpte2085 (11), chroma-derived-nc (12), chroma-derived-c (13), ictcp (14)
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setColorspace(int value);
    int getColorspace() const;

    /**
     * select color range
     * Unit: range
     * Possible Values: unspecified (0), unknown (0), limited (1), tv (1), mpeg (1), full (2), pc (2), jpeg (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setRange(int value);
    int getRange() const;

    Buffer(int height = 0, std::pair<int, int> video_size = std::make_pair<int, int>(0, 1), const std::string& pix_fmt = "", std::pair<int, int> pixel_aspect = std::make_pair<int, int>(0, 1), std::pair<int, int> frame_rate = std::make_pair<int, int>(0, 1), int colorspace = 2, int range = 0);
    virtual ~Buffer();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int height_;
    std::pair<int, int> video_size_;
    std::string pix_fmt_;
    std::pair<int, int> pixel_aspect_;
    std::pair<int, int> frame_rate_;
    int colorspace_;
    int range_;
};
