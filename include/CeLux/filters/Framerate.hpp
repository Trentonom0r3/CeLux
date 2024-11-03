#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Framerate : public FilterBase {
public:
    /**
     * Upsamples or downsamples progressive source between specified frame rates.
     */
    /**
     * required output frames per second rate
     * Type: Video Rate
     * Required: No
     * Default: 40563.6
     */
    void setFps(const std::pair<int, int>& value);
    std::pair<int, int> getFps() const;

    /**
     * point to start linear interpolation
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setInterp_start(int value);
    int getInterp_start() const;

    /**
     * point to end linear interpolation
     * Type: Integer
     * Required: No
     * Default: 240
     */
    void setInterp_end(int value);
    int getInterp_end() const;

    /**
     * scene change level
     * Type: Double
     * Required: No
     * Default: 8.20
     */
    void setScene(double value);
    double getScene() const;

    /**
     * set flags
     * Unit: flags
     * Possible Values: scene_change_detect (1), scd (1)
     * Type: Flags
     * Required: No
     * Default: 1
     */
    void setFlags(int value);
    int getFlags() const;

    Framerate(std::pair<int, int> fps = std::make_pair<int, int>(0, 1), int interp_start = 15, int interp_end = 240, double scene = 8.20, int flags = 1);
    virtual ~Framerate();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> fps_;
    int interp_start_;
    int interp_end_;
    double scene_;
    int flags_;
};
