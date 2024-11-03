#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Colorspectrum : public FilterBase {
public:
    /**
     * Generate colors spectrum.
     */
    /**
     * set video size
     * Aliases: s
     * Type: Image Size
     * Required: No
     * Default: 320x240
     */
    void setSize(const std::pair<int, int>& value);
    std::pair<int, int> getSize() const;

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
     * set the color spectrum type
     * Unit: type
     * Possible Values: black (0), white (1), all (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setType(int value);
    int getType() const;

    Colorspectrum(std::pair<int, int> size = std::make_pair<int, int>(0, 1), std::pair<int, int> rate = std::make_pair<int, int>(0, 1), int64_t duration = 0, std::pair<int, int> sar = std::make_pair<int, int>(0, 1), int type = 0);
    virtual ~Colorspectrum();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> size_;
    std::pair<int, int> rate_;
    int64_t duration_;
    std::pair<int, int> sar_;
    int type_;
};
