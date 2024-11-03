#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Siti : public FilterBase {
public:
    /**
     * Calculate spatial information (SI) and temporal information (TI).
     */
    /**
     * Print summary showing average values
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setPrint_summary(bool value);
    bool getPrint_summary() const;

    Siti(bool print_summary = false);
    virtual ~Siti();

    std::string getFilterDescription() const override;

private:
    // Option variables
    bool print_summary_;
};
