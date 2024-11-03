#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Vmafmotion : public FilterBase {
public:
    /**
     * Calculate the VMAF Motion score.
     */
    /**
     * Set file where to store per-frame difference information
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setStats_file(const std::string& value);
    std::string getStats_file() const;

    Vmafmotion(const std::string& stats_file = "");
    virtual ~Vmafmotion();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string stats_file_;
};
