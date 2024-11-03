#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Psnr : public FilterBase {
public:
    /**
     * Calculate the PSNR between two video streams.
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

    /**
     * Set the format version for the stats file.
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setStats_version(int value);
    int getStats_version() const;

    /**
     * Add raw stats (max values) to the output log.
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setOutput_max(bool value);
    bool getOutput_max() const;

    Psnr(const std::string& stats_file = "", int stats_version = 1, bool output_max = false);
    virtual ~Psnr();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string stats_file_;
    int stats_version_;
    bool output_max_;
};
