#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Scdet : public FilterBase {
public:
    /**
     * Detect video scene change
     */
    /**
     * set scene change detect threshold
     * Aliases: t
     * Type: Double
     * Required: No
     * Default: 10.00
     */
    void setThreshold(double value);
    double getThreshold() const;

    /**
     * Set the flag to pass scene change frames
     * Aliases: s
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setSc_pass(bool value);
    bool getSc_pass() const;

    Scdet(double threshold = 10.00, bool sc_pass = false);
    virtual ~Scdet();

    std::string getFilterDescription() const override;

private:
    // Option variables
    double threshold_;
    bool sc_pass_;
};
