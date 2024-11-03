#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Colorchart : public FilterBase {
public:
    /**
     * Generate color checker chart.
     */
    /**
     * set video rate
     * Aliases: r
     * Type: Video Rate
     * Required: No
     * Default: 40439.8
     */
    void setRate(const std::pair<int, int>& value);
    std::pair<int, int> getRate() const;

    /**
     * set video duration
     * Aliases: d
     * Type: Duration
     * Required: No
     * Default: -1
     */
    void setDuration(int64_t value);
    int64_t getDuration() const;

    /**
     * set video sample aspect ratio
     * Type: Rational
     * Required: No
     * Default: 0
     */
    void setSar(const std::pair<int, int>& value);
    std::pair<int, int> getSar() const;

    /**
     * set the single patch size
     * Type: Image Size
     * Required: No
     * Default: 64x64
     */
    void setPatch_size(const std::pair<int, int>& value);
    std::pair<int, int> getPatch_size() const;

    /**
     * set the color checker chart preset
     * Unit: preset
     * Possible Values: reference (0), skintones (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setPreset(int value);
    int getPreset() const;

    Colorchart(std::pair<int, int> rate = std::make_pair<int, int>(0, 1), int64_t duration = 0, std::pair<int, int> sar = std::make_pair<int, int>(0, 1), std::pair<int, int> patch_size = std::make_pair<int, int>(0, 1), int preset = 0);
    virtual ~Colorchart();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> rate_;
    int64_t duration_;
    std::pair<int, int> sar_;
    std::pair<int, int> patch_size_;
    int preset_;
};
