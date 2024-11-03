#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Readvitc : public FilterBase {
public:
    /**
     * Read vertical interval timecode and write it to frame metadata.
     */
    /**
     * maximum line numbers to scan for VITC data
     * Type: Integer
     * Required: No
     * Default: 45
     */
    void setScan_max(int value);
    int getScan_max() const;

    /**
     * black color threshold
     * Type: Double
     * Required: No
     * Default: 0.20
     */
    void setThr_b(double value);
    double getThr_b() const;

    /**
     * white color threshold
     * Type: Double
     * Required: No
     * Default: 0.60
     */
    void setThr_w(double value);
    double getThr_w() const;

    Readvitc(int scan_max = 45, double thr_b = 0.20, double thr_w = 0.60);
    virtual ~Readvitc();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int scan_max_;
    double thr_b_;
    double thr_w_;
};
