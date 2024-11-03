#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Trim : public FilterBase {
public:
    /**
     * Pick one continuous section from the input, drop the rest.
     */
    /**
     * Timestamp of the first frame that should be passed
     * Aliases: start
     * Type: Duration
     * Required: No
     * Default: 9223372036854775807
     */
    void setStarti(int64_t value);
    int64_t getStarti() const;

    /**
     * Timestamp of the first frame that should be dropped again
     * Aliases: end
     * Type: Duration
     * Required: No
     * Default: 9223372036854775807
     */
    void setEndi(int64_t value);
    int64_t getEndi() const;

    /**
     * Timestamp of the first frame that should be  passed
     * Type: Integer64
     * Required: No
     * Default: -9223372036854775808
     */
    void setStart_pts(int64_t value);
    int64_t getStart_pts() const;

    /**
     * Timestamp of the first frame that should be dropped again
     * Type: Integer64
     * Required: No
     * Default: -9223372036854775808
     */
    void setEnd_pts(int64_t value);
    int64_t getEnd_pts() const;

    /**
     * Maximum duration of the output
     * Aliases: duration
     * Type: Duration
     * Required: No
     * Default: 0
     */
    void setDurationi(int64_t value);
    int64_t getDurationi() const;

    /**
     * Number of the first frame that should be passed to the output
     * Type: Integer64
     * Required: No
     * Default: -1
     */
    void setStart_frame(int64_t value);
    int64_t getStart_frame() const;

    /**
     * Number of the first frame that should be dropped again
     * Type: Integer64
     * Required: No
     * Default: 9223372036854775807
     */
    void setEnd_frame(int64_t value);
    int64_t getEnd_frame() const;

    Trim(int64_t starti = 9223372036854775807ULL, int64_t endi = 9223372036854775807ULL, int64_t start_pts = 0, int64_t end_pts = 0, int64_t durationi = 0ULL, int64_t start_frame = 0, int64_t end_frame = 9223372036854775807ULL);
    virtual ~Trim();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int64_t starti_;
    int64_t endi_;
    int64_t start_pts_;
    int64_t end_pts_;
    int64_t durationi_;
    int64_t start_frame_;
    int64_t end_frame_;
};
