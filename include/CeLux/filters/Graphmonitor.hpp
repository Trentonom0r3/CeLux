#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Graphmonitor : public FilterBase {
public:
    /**
     * Show various filtergraph stats.
     */
    /**
     * set monitor size
     * Aliases: s
     * Type: Image Size
     * Required: No
     * Default: hd720
     */
    void setSize(const std::pair<int, int>& value);
    std::pair<int, int> getSize() const;

    /**
     * set video opacity
     * Aliases: o
     * Type: Float
     * Required: No
     * Default: 0.90
     */
    void setOpacity(float value);
    float getOpacity() const;

    /**
     * set mode
     * Aliases: m
     * Unit: mode
     * Possible Values: full (0), compact (1), nozero (2), noeof (4), nodisabled (8)
     * Type: Flags
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    /**
     * set flags
     * Aliases: f
     * Unit: flags
     * Possible Values: none (0), all (2147483647), queue (1), frame_count_in (4), frame_count_out (2), frame_count_delta (16384), pts (8), pts_delta (4096), time (16), time_delta (8192), timebase (32), format (64), size (128), rate (256), eof (512), sample_count_in (2048), sample_count_out (1024), sample_count_delta (32768), disabled (65536)
     * Type: Flags
     * Required: No
     * Default: 1
     */
    void setFlags(int value);
    int getFlags() const;

    /**
     * set video rate
     * Aliases: r
     * Type: Video Rate
     * Required: No
     * Default: 40439.8
     */
    void setRate(const std::pair<int, int>& value);
    std::pair<int, int> getRate() const;

    Graphmonitor(std::pair<int, int> size = std::make_pair<int, int>(0, 1), float opacity = 0.90, int mode = 0, int flags = 1, std::pair<int, int> rate = std::make_pair<int, int>(0, 1));
    virtual ~Graphmonitor();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> size_;
    float opacity_;
    int mode_;
    int flags_;
    std::pair<int, int> rate_;
};
