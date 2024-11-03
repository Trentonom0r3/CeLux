#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Idet : public FilterBase {
public:
    /**
     * Interlace detect Filter.
     */
    /**
     * set interlacing threshold
     * Type: Float
     * Required: No
     * Default: 1.04
     */
    void setIntl_thres(float value);
    float getIntl_thres() const;

    /**
     * set progressive threshold
     * Type: Float
     * Required: No
     * Default: 1.50
     */
    void setProg_thres(float value);
    float getProg_thres() const;

    /**
     * set repeat threshold
     * Type: Float
     * Required: No
     * Default: 3.00
     */
    void setRep_thres(float value);
    float getRep_thres() const;

    /**
     * half life of cumulative statistics
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setHalf_life(float value);
    float getHalf_life() const;

    /**
     * set number of frames to use to determine if the interlace flag is accurate
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setAnalyze_interlaced_flag(int value);
    int getAnalyze_interlaced_flag() const;

    Idet(float intl_thres = 1.04, float prog_thres = 1.50, float rep_thres = 3.00, float half_life = 0.00, int analyze_interlaced_flag = 0);
    virtual ~Idet();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float intl_thres_;
    float prog_thres_;
    float rep_thres_;
    float half_life_;
    int analyze_interlaced_flag_;
};
