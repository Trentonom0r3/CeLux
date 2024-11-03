#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Ssim : public FilterBase {
public:
    /**
     * Calculate the SSIM between two video streams.
     */
    /**
     * Set file where to store per-frame difference information
     * Aliases: f
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setStats_file(const std::string& value);
    std::string getStats_file() const;

    Ssim(const std::string& stats_file = "");
    virtual ~Ssim();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string stats_file_;
};
