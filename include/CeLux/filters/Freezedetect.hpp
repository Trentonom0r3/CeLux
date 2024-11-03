#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Freezedetect : public FilterBase {
public:
    /**
     * Detects frozen video input.
     */
    /**
     * set noise tolerance
     * Aliases: n
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setNoise(double value);
    double getNoise() const;

    /**
     * set minimum duration in seconds
     * Aliases: d
     * Type: Duration
     * Required: No
     * Default: 2000000
     */
    void setDuration(int64_t value);
    int64_t getDuration() const;

    Freezedetect(double noise = 0.00, int64_t duration = 2000000ULL);
    virtual ~Freezedetect();

    std::string getFilterDescription() const override;

private:
    // Option variables
    double noise_;
    int64_t duration_;
};
